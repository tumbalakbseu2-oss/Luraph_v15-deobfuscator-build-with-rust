//! High-Level Region Structuring & Luau Source Reconstruction.
//!
//! Reconstructs hierarchical structured regions (`Block`, `If`, `Loop`, `Seq`,
//! `Break`, `Continue`, `Return`) from unstructured CFGs, resolves branch merge
//! points using post-dominators, unifies loop exits, and emits clean Luau source code.

use std::collections::HashSet;

use crate::cfg::dominator::{DominatorTree, LoopExtractor, NaturalLoop, PostDominatorTree};
use crate::cfg::ir::{
    BlockId, ControlFlowGraph, IrExpr, IrInst, LoopKind, StructuredRegion, Terminator, UnOp,
};

/// High-level region structurer.
pub struct RegionStructurer<'a, 'arena> {
    pub cfg: &'a ControlFlowGraph<'arena>,
    pub dom: DominatorTree,
    pub pdom: PostDominatorTree,
    pub loops: Vec<NaturalLoop>,
    pub emitted: HashSet<BlockId>,
}

impl<'a, 'arena> RegionStructurer<'a, 'arena> {
    pub fn new(cfg: &'a ControlFlowGraph<'arena>) -> Self {
        let dom = DominatorTree::compute(cfg);
        let pdom = PostDominatorTree::compute(cfg);
        let loops = LoopExtractor::extract_loops(cfg, &dom);

        Self {
            cfg,
            dom,
            pdom,
            loops,
            emitted: HashSet::new(),
        }
    }

    /// Structures the entire CFG into a single root `StructuredRegion`.
    pub fn structure(&mut self) -> StructuredRegion<'arena> {
        if self.cfg.blocks.is_empty() {
            return StructuredRegion::Seq(Vec::new());
        }

        let entry = self.cfg.entry;
        self.structure_region(entry, None, None)
    }

    /// Recursively structures a region starting from `cur` until hitting boundary `stop`.
    fn structure_region(
        &mut self,
        mut cur: BlockId,
        stop: Option<BlockId>,
        loop_ctx: Option<&NaturalLoop>,
    ) -> StructuredRegion<'arena> {
        let mut regions = Vec::new();

        while cur.is_valid() && Some(cur) != stop {
            if self.emitted.contains(&cur) {
                // If we reach a loop header, emit Continue
                if let Some(l) = loop_ctx {
                    if cur == l.header {
                        regions.push(StructuredRegion::Continue);
                        break;
                    }
                }
                break;
            }

            // Check if current block is a loop exit target
            if let Some(l) = loop_ctx {
                if !l.body.contains(&cur) {
                    regions.push(StructuredRegion::Break);
                    break;
                }
            }

            // Check if current block is a natural loop header
            if let Some(nat_loop) = self.loops.iter().find(|l| l.header == cur).cloned() {
                let loop_region = self.structure_loop(&nat_loop);
                regions.push(loop_region);

                // Advance to loop exit block if known
                if let Some(&(_, exit_target)) = nat_loop.exits.first() {
                    cur = exit_target;
                    continue;
                } else {
                    break;
                }
            }

            let Some(block) = self.cfg.blocks.get(&cur) else {
                break;
            };

            self.emitted.insert(cur);
            regions.push(StructuredRegion::Block(cur));

            match &block.terminator {
                Terminator::Return(vals) => {
                    regions.push(StructuredRegion::Return(vals.to_vec()));
                    break;
                }
                Terminator::Crash => {
                    regions.push(StructuredRegion::Crash);
                    break;
                }
                Terminator::Unreachable => {
                    break;
                }
                Terminator::Goto(target) => {
                    if Some(*target) == stop {
                        break;
                    }
                    if let Some(l) = loop_ctx {
                        if *target == l.header {
                            regions.push(StructuredRegion::Continue);
                            break;
                        }
                        if !l.body.contains(target) {
                            regions.push(StructuredRegion::Break);
                            break;
                        }
                    }
                    cur = *target;
                }
                Terminator::Branch {
                    cond,
                    true_target,
                    false_target,
                } => {
                    // Resolve merge join block: immediate post-dominator or stop boundary
                    let merge_block = self
                        .pdom
                        .ipdom
                        .get(&cur)
                        .copied()
                        .filter(|&m| m != cur && m != *true_target && m != *false_target);

                    let then_region = self.structure_region(*true_target, merge_block.or(stop), loop_ctx);
                    let else_region = self.structure_region(*false_target, merge_block.or(stop), loop_ctx);

                    let else_branch = if is_empty_region(&else_region) {
                        None
                    } else {
                        Some(Box::new(else_region))
                    };

                    regions.push(StructuredRegion::If {
                        cond: cond.clone(),
                        then_branch: Box::new(then_region),
                        else_branch,
                    });

                    if let Some(m) = merge_block {
                        cur = m;
                    } else {
                        break;
                    }
                }
                Terminator::ForLoop {
                    var,
                    start,
                    limit,
                    step,
                    body,
                    exit,
                } => {
                    let body_region = self.structure_region(*body, Some(*exit), loop_ctx);
                    regions.push(StructuredRegion::Loop {
                        kind: LoopKind::Numeric {
                            var: *var,
                            start: start.clone(),
                            limit: limit.clone(),
                            step: step.clone(),
                        },
                        body: Box::new(body_region),
                    });
                    cur = *exit;
                }
                Terminator::ForInLoop {
                    vars,
                    iterators,
                    body,
                    exit,
                } => {
                    let body_region = self.structure_region(*body, Some(*exit), loop_ctx);
                    regions.push(StructuredRegion::Loop {
                        kind: LoopKind::Generic {
                            vars: vars.clone(),
                            iterators: iterators.clone(),
                        },
                        body: Box::new(body_region),
                    });
                    cur = *exit;
                }
            }
        }

        if regions.len() == 1 {
            regions.pop().unwrap()
        } else {
            StructuredRegion::Seq(regions)
        }
    }

    /// Structures a natural loop body.
    fn structure_loop(&mut self, nat_loop: &NaturalLoop) -> StructuredRegion<'arena> {
        let header = nat_loop.header;
        self.emitted.insert(header);

        let header_block = self.cfg.blocks.get(&header);
        let (loop_kind, body_start) = match header_block.map(|b| &b.terminator) {
            Some(Terminator::Branch {
                cond,
                true_target,
                false_target,
            }) => {
                // If true branch stays in loop, loop is `while cond do`
                if nat_loop.body.contains(true_target) {
                    (LoopKind::While(cond.clone()), *true_target)
                } else {
                    // Inverted condition: while not cond do
                    let negated = IrExpr::Unary {
                        op: UnOp::Not,
                        expr: Box::leak(Box::new(cond.clone())),
                    };
                    (LoopKind::While(negated), *false_target)
                }
            }
            Some(Terminator::Goto(target)) => (
                LoopKind::While(IrExpr::Boolean(true)),
                *target,
            ),
            _ => (
                LoopKind::While(IrExpr::Boolean(true)),
                header,
            ),
        };

        let body_region = self.structure_region(body_start, Some(header), Some(nat_loop));

        StructuredRegion::Loop {
            kind: loop_kind,
            body: Box::new(body_region),
        }
    }
}

fn is_empty_region(region: &StructuredRegion) -> bool {
    match region {
        StructuredRegion::Seq(v) => v.iter().all(is_empty_region),
        _ => false,
    }
}

/// Emits idiomatic, syntactically valid Luau source code from a StructuredRegion.
pub fn emit_luau(
    region: &StructuredRegion,
    cfg: &ControlFlowGraph,
    indent: usize,
) -> String {
    let mut out = String::new();
    emit_region(region, cfg, indent, &mut out);
    out
}

fn emit_indent(indent: usize, out: &mut String) {
    for _ in 0..indent {
        out.push_str("    ");
    }
}

fn emit_region(
    region: &StructuredRegion,
    cfg: &ControlFlowGraph,
    indent: usize,
    out: &mut String,
) {
    match region {
        StructuredRegion::Block(bid) => {
            if let Some(block) = cfg.blocks.get(bid) {
                for inst in &block.instructions {
                    emit_inst(inst, indent, out);
                }
            }
        }
        StructuredRegion::If {
            cond,
            then_branch,
            else_branch,
        } => {
            emit_indent(indent, out);
            out.push_str("if ");
            out.push_str(&expr_to_string(cond));
            out.push_str(" then\n");
            emit_region(then_branch, cfg, indent + 1, out);

            if let Some(else_b) = else_branch {
                emit_indent(indent, out);
                out.push_str("else\n");
                emit_region(else_b, cfg, indent + 1, out);
            }

            emit_indent(indent, out);
            out.push_str("end\n");
        }
        StructuredRegion::Loop { kind, body } => {
            if is_empty_region(body) {
                return;
            }
            emit_indent(indent, out);
            match kind {
                LoopKind::While(cond) => {
                    out.push_str("while ");
                    out.push_str(&expr_to_string(cond));
                    out.push_str(" do\n");
                }
                LoopKind::Repeat(cond) => {
                    out.push_str("repeat\n");
                    emit_region(body, cfg, indent + 1, out);
                    emit_indent(indent, out);
                    out.push_str("until ");
                    out.push_str(&expr_to_string(cond));
                    out.push('\n');
                    return;
                }
                LoopKind::Numeric {
                    var,
                    start,
                    limit,
                    step,
                } => {
                    out.push_str(&format!(
                        "for r{} = {}, {}",
                        var,
                        expr_to_string(start),
                        expr_to_string(limit)
                    ));
                    if let Some(s) = step {
                        out.push_str(&format!(", {}", expr_to_string(s)));
                    }
                    out.push_str(" do\n");
                }
                LoopKind::Generic { vars, iterators } => {
                    let var_names: Vec<String> = vars.iter().map(|v| format!("r{v}")).collect();
                    let iters: Vec<String> = iterators.iter().map(expr_to_string).collect();
                    out.push_str(&format!(
                        "for {} in {} do\n",
                        var_names.join(", "),
                        iters.join(", ")
                    ));
                }
            }
            emit_region(body, cfg, indent + 1, out);
            emit_indent(indent, out);
            out.push_str("end\n");
        }
        StructuredRegion::Seq(regions) => {
            for r in regions {
                emit_region(r, cfg, indent, out);
            }
        }
        StructuredRegion::Break => {
            emit_indent(indent, out);
            out.push_str("break\n");
        }
        StructuredRegion::Continue => {
            emit_indent(indent, out);
            out.push_str("continue\n");
        }
        StructuredRegion::Return(vals) => {
            emit_indent(indent, out);
            if vals.is_empty() {
                out.push_str("return\n");
            } else {
                let val_strs: Vec<String> = vals.iter().map(expr_to_string).collect();
                out.push_str(&format!("return {}\n", val_strs.join(", ")));
            }
        }
        StructuredRegion::Crash => {
            emit_indent(indent, out);
            out.push_str("error(\"LPH_CRASH\")\n");
        }
        StructuredRegion::Error(msg) => {
            emit_indent(indent, out);
            out.push_str(&format!("error(\"{msg}\")\n"));
        }
    }
}

fn emit_inst(inst: &IrInst, indent: usize, out: &mut String) {
    match inst {
        IrInst::Assign { dest, src } => {
            emit_indent(indent, out);
            out.push_str(&format!("local r{} = {}\n", dest, expr_to_string(src)));
        }
        IrInst::TableSet { table, key, value } => {
            emit_indent(indent, out);
            let tbl_str = expr_to_string(table);
            if tbl_str.starts_with('(') {
                out.push(';');
            }
            out.push_str(&format!(
                "{}[{}] = {}\n",
                tbl_str,
                expr_to_string(key),
                expr_to_string(value)
            ));
        }
        IrInst::SetUpvalue { upvalue_idx, src } => {
            emit_indent(indent, out);
            out.push_str(&format!("up_{} = {}\n", upvalue_idx, expr_to_string(src)));
        }
        IrInst::SetGlobal { name, src } => {
            emit_indent(indent, out);
            if crate::codegen::is_valid_ident(name) {
                out.push_str(&format!("{} = {}\n", name, expr_to_string(src)));
            } else {
                out.push_str(&format!("_G[\"{}\"] = {}\n", name.escape_default(), expr_to_string(src)));
            }
        }
        IrInst::CallStmt {
            callee,
            args,
            ret_dest,
            ..
        } => {
            emit_indent(indent, out);
            let arg_strs: Vec<String> = args.iter().map(expr_to_string).collect();
            let callee_str = expr_to_string(callee);
            let call_expr = if ret_dest.is_none() && callee_str.starts_with('(') {
                format!(";{}({})", callee_str, arg_strs.join(", "))
            } else {
                format!("{}({})", callee_str, arg_strs.join(", "))
            };
            if let Some(dest) = ret_dest {
                out.push_str(&format!("local r{} = {}\n", dest, call_expr));
            } else {
                out.push_str(&format!("{}\n", call_expr));
            }
        }

        IrInst::SetList {
            table,
            start_idx,
            values,
        } => {
            emit_indent(indent, out);
            for (offset, val) in values.iter().enumerate() {
                out.push_str(&format!(
                    "{}[{}] = {}\n",
                    expr_to_string(table),
                    start_idx + offset as u32,
                    expr_to_string(val)
                ));
            }
        }
        IrInst::CloseUpvalue(reg) => {
            emit_indent(indent, out);
            out.push_str(&format!("-- close upvalues >= r{}\n", reg));
        }
        IrInst::Comment(c) => {
            emit_indent(indent, out);
            out.push_str(&format!("-- {}\n", c));
        }
        IrInst::Nop => {}
    }
}

pub fn expr_to_string(expr: &IrExpr) -> String {
    match expr {
        IrExpr::Nil => "nil".to_string(),
        IrExpr::Boolean(b) => b.to_string(),
        IrExpr::Number(n) => {
            if n.fract() == 0.0 && n.is_finite() {
                format!("{:.0}", n)
            } else {
                format!("{n}")
            }
        }
        IrExpr::String(s) => format!("\"{}\"", s.escape_default()),
        IrExpr::Register(r) => format!("r{r}"),
        IrExpr::Upvalue(u) => format!("up_{u}"),
        IrExpr::Global(g) => {
            if crate::codegen::is_valid_ident(g) {
                g.to_string()
            } else {
                format!("_G[\"{}\"]", g.escape_default())
            }
        }
        IrExpr::Constant(k) => format!("_k[{k}]"),
        IrExpr::Unary { op, expr } => format!("({}{})", op, expr_to_string(expr)),
        IrExpr::Binary { op, left, right } => format!(
            "({} {} {})",
            expr_to_string(left),
            op,
            expr_to_string(right)
        ),
        IrExpr::TableGet { table, key } => {
            format!("{}[{}]", expr_to_string(table), expr_to_string(key))
        }
        IrExpr::Call { callee, args } => {
            let arg_strs: Vec<String> = args.iter().map(expr_to_string).collect();
            format!("{}({})", expr_to_string(callee), arg_strs.join(", "))
        }
        IrExpr::NewTable { .. } => "{}".to_string(),
        IrExpr::Closure { proto_id, .. } => format!("_protos[{proto_id}]"),
        IrExpr::Vararg => "...".to_string(),
        IrExpr::IfExp {
            cond,
            then_expr,
            else_expr,
        } => format!(
            "(if {} then {} else {})",
            expr_to_string(cond),
            expr_to_string(then_expr),
            expr_to_string(else_expr)
        ),
        IrExpr::Phi(_) => "phi()".to_string(),
    }
}

/// Lowers an IR expression into a codegen LuauExpr, resolving variable names via SsaContext.
pub fn lower_expr<'arena>(
    expr: &IrExpr<'arena>,
    bid: Option<BlockId>,
    inst_idx: Option<usize>,
    ssa: &crate::ssa::SsaContext,
    bump: &'arena bumpalo::Bump,
) -> crate::codegen::LuauExpr<'arena> {
    match expr {
        IrExpr::Nil => crate::codegen::LuauExpr::Nil,
        IrExpr::Boolean(b) => crate::codegen::LuauExpr::Boolean(*b),
        IrExpr::Number(n) => crate::codegen::LuauExpr::Number(*n),
        IrExpr::String(s) => crate::codegen::LuauExpr::String(s),
        IrExpr::Register(r) => {
            let mut resolved_name: Option<&str> = None;
            if let (Some(b), Some(idx)) = (bid, inst_idx) {
                let site = crate::ssa::UseSite::inst(b, idx, *r);
                if let Some(Some(web_id)) = ssa.use_to_web.get(&site) {
                    if let Some(web) = ssa.get_web(*web_id) {
                        if !web.assigned_name.is_empty() {
                            resolved_name = Some(&web.assigned_name);
                        }
                    }
                }
            }
            if resolved_name.is_none() {
                if let Some(web) = ssa.webs.iter().find(|w| w.original_reg == *r && !w.assigned_name.is_empty()) {
                    resolved_name = Some(&web.assigned_name);
                }
            }
            let name = match resolved_name {
                Some(n) => bump.alloc_str(n),
                None => bump.alloc_str(&format!("r{r}")),
            };
            crate::codegen::LuauExpr::Identifier(name)
        }
        IrExpr::Upvalue(u) => {
            let name = bump.alloc_str(&format!("up_{u}"));
            crate::codegen::LuauExpr::Identifier(name)
        }
        IrExpr::Global(g) => crate::codegen::LuauExpr::Identifier(g),
        IrExpr::Constant(k) => crate::codegen::LuauExpr::Index {
            table: Box::new(crate::codegen::LuauExpr::Identifier("_k")),
            key: Box::new(crate::codegen::LuauExpr::Number(*k as f64)),
        },
        IrExpr::Unary { op, expr } => {
            let inner = lower_expr(expr, bid, inst_idx, ssa, bump);
            crate::codegen::LuauExpr::Unary {
                op: (*op).into(),
                expr: Box::new(inner),
            }
        }
        IrExpr::Binary { op, left, right } => {
            let l = lower_expr(left, bid, inst_idx, ssa, bump);
            let r = lower_expr(right, bid, inst_idx, ssa, bump);
            crate::codegen::LuauExpr::Binary {
                op: (*op).into(),
                left: Box::new(l),
                right: Box::new(r),
            }
        }
        IrExpr::TableGet { table, key } => {
            let t = lower_expr(table, bid, inst_idx, ssa, bump);
            let k = lower_expr(key, bid, inst_idx, ssa, bump);
            crate::codegen::LuauExpr::Index {
                table: Box::new(t),
                key: Box::new(k),
            }
        }
        IrExpr::Call { callee, args } => {
            let f = lower_expr(callee, bid, inst_idx, ssa, bump);
            let arg_exprs = args.iter().map(|a| lower_expr(a, bid, inst_idx, ssa, bump)).collect();
            crate::codegen::LuauExpr::Call {
                func: Box::new(f),
                args: arg_exprs,
            }
        }
        IrExpr::NewTable { .. } => crate::codegen::LuauExpr::Table(Vec::new()),
        IrExpr::Closure { proto_id, .. } => {
            let table = Box::new(crate::codegen::LuauExpr::Identifier(bump.alloc_str("_protos")));
            let key = Box::new(crate::codegen::LuauExpr::Number(*proto_id as f64));
            crate::codegen::LuauExpr::Index { table, key }
        }
        IrExpr::Vararg => crate::codegen::LuauExpr::Vararg,
        IrExpr::IfExp { cond, then_expr, else_expr } => {
            let c = lower_expr(cond, bid, inst_idx, ssa, bump);
            let t = lower_expr(then_expr, bid, inst_idx, ssa, bump);
            let e = lower_expr(else_expr, bid, inst_idx, ssa, bump);
            let raw = bump.alloc_str(&format!("(if {} then {} else {})", c.to_code(), t.to_code(), e.to_code()));
            crate::codegen::LuauExpr::Raw(raw)
        }
        IrExpr::Phi(_) => crate::codegen::LuauExpr::Raw("phi()"),
    }
}

/// Lowers an IR instruction into codegen LuauStats.
pub fn lower_inst<'arena>(
    inst: &IrInst<'arena>,
    bid: BlockId,
    inst_idx: usize,
    ssa: &crate::ssa::SsaContext,
    bump: &'arena bumpalo::Bump,
) -> Vec<crate::codegen::LuauStat<'arena>> {
    let mut out = Vec::new();
    match inst {
        IrInst::Assign { dest, src } => {
            let def_site = crate::ssa::DefSite::inst(bid, inst_idx, *dest);
            let name = if let Some(&web_id) = ssa.def_to_web.get(&def_site) {
                if let Some(web) = ssa.get_web(web_id) {
                    if !web.assigned_name.is_empty() {
                        bump.alloc_str(&web.assigned_name)
                    } else {
                        bump.alloc_str(&format!("r{dest}"))
                    }
                } else {
                    bump.alloc_str(&format!("r{dest}"))
                }
            } else {
                bump.alloc_str(&format!("r{dest}"))
            };

            let val = lower_expr(src, Some(bid), Some(inst_idx), ssa, bump);
            out.push(crate::codegen::LuauStat::LocalAssign {
                names: vec![name],
                values: vec![val],
            });
        }
        IrInst::TableSet { table, key, value } => {
            let t = lower_expr(table, Some(bid), Some(inst_idx), ssa, bump);
            let k = lower_expr(key, Some(bid), Some(inst_idx), ssa, bump);
            let v = lower_expr(value, Some(bid), Some(inst_idx), ssa, bump);
            out.push(crate::codegen::LuauStat::Assign {
                targets: vec![crate::codegen::LuauExpr::Index {
                    table: Box::new(t),
                    key: Box::new(k),
                }],
                values: vec![v],
            });
        }
        IrInst::SetUpvalue { upvalue_idx, src } => {
            let name = bump.alloc_str(&format!("up_{upvalue_idx}"));
            let v = lower_expr(src, Some(bid), Some(inst_idx), ssa, bump);
            out.push(crate::codegen::LuauStat::Assign {
                targets: vec![crate::codegen::LuauExpr::Identifier(name)],
                values: vec![v],
            });
        }
        IrInst::SetGlobal { name, src } => {
            let v = lower_expr(src, Some(bid), Some(inst_idx), ssa, bump);
            out.push(crate::codegen::LuauStat::Assign {
                targets: vec![crate::codegen::LuauExpr::Identifier(name)],
                values: vec![v],
            });
        }
        IrInst::CallStmt {
            callee,
            args,
            ret_dest,
            ..
        } => {
            let f = lower_expr(callee, Some(bid), Some(inst_idx), ssa, bump);
            let arg_exprs = args
                .iter()
                .map(|a| lower_expr(a, Some(bid), Some(inst_idx), ssa, bump))
                .collect();
            let call_expr = crate::codegen::LuauExpr::Call {
                func: Box::new(f),
                args: arg_exprs,
            };

            if let Some(dest) = ret_dest {
                let def_site = crate::ssa::DefSite::inst(bid, inst_idx, *dest);
                let name = if let Some(&web_id) = ssa.def_to_web.get(&def_site) {
                    if let Some(web) = ssa.get_web(web_id) {
                        if !web.assigned_name.is_empty() {
                            bump.alloc_str(&web.assigned_name)
                        } else {
                            bump.alloc_str(&format!("r{dest}"))
                        }
                    } else {
                        bump.alloc_str(&format!("r{dest}"))
                    }
                } else {
                    bump.alloc_str(&format!("r{dest}"))
                };

                out.push(crate::codegen::LuauStat::LocalAssign {
                    names: vec![name],
                    values: vec![call_expr],
                });
            } else {
                out.push(crate::codegen::LuauStat::Call(call_expr));
            }
        }
        IrInst::SetList {
            table,
            start_idx,
            values,
        } => {
            let t = lower_expr(table, Some(bid), Some(inst_idx), ssa, bump);
            for (offset, val) in values.iter().enumerate() {
                let v = lower_expr(val, Some(bid), Some(inst_idx), ssa, bump);
                let k = crate::codegen::LuauExpr::Number((*start_idx + offset as u32) as f64);
                out.push(crate::codegen::LuauStat::Assign {
                    targets: vec![crate::codegen::LuauExpr::Index {
                        table: Box::new(t.clone()),
                        key: Box::new(k),
                    }],
                    values: vec![v],
                });
            }
        }
        IrInst::CloseUpvalue(reg) => {
            let comment = bump.alloc_str(&format!("close upvalues >= r{reg}"));
            out.push(crate::codegen::LuauStat::Comment(comment));
        }
        IrInst::Comment(c) => {
            out.push(crate::codegen::LuauStat::Comment(c));
        }
        IrInst::Nop => {}
    }
    out
}

/// Recursively lowers a StructuredRegion into a vector of LuauStat.
pub fn lower_region<'arena>(
    region: &StructuredRegion<'arena>,
    cfg: &ControlFlowGraph<'arena>,
    ssa: &crate::ssa::SsaContext,
    bump: &'arena bumpalo::Bump,
) -> Vec<crate::codegen::LuauStat<'arena>> {
    let mut out = Vec::new();
    match region {
        StructuredRegion::Block(bid) => {
            if let Some(block) = cfg.blocks.get(bid) {
                for (idx, inst) in block.instructions.iter().enumerate() {
                    out.extend(lower_inst(inst, *bid, idx, ssa, bump));
                }
            }
        }
        StructuredRegion::If {
            cond,
            then_branch,
            else_branch,
        } => {
            let c = lower_expr(cond, None, None, ssa, bump);
            let then_stats = lower_region(then_branch, cfg, ssa, bump);
            let else_stats = else_branch.as_ref().map(|e| lower_region(e, cfg, ssa, bump));
            out.push(crate::codegen::LuauStat::If {
                cond: c,
                then_branch: then_stats,
                elseif_branches: Vec::new(),
                else_branch: else_stats,
            });
        }
        StructuredRegion::Loop { kind, body } => {
            let body_stats = lower_region(body, cfg, ssa, bump);
            match kind {
                LoopKind::While(cond) => {
                    if !body_stats.is_empty() {
                        let c = lower_expr(cond, None, None, ssa, bump);
                        out.push(crate::codegen::LuauStat::While {
                            cond: c,
                            body: body_stats,
                        });
                    }
                }
                LoopKind::Repeat(cond) => {
                    if !body_stats.is_empty() {
                        let c = lower_expr(cond, None, None, ssa, bump);
                        out.push(crate::codegen::LuauStat::Repeat {
                            cond: c,
                            body: body_stats,
                        });
                    }
                }
                LoopKind::Numeric {
                    var,
                    start,
                    limit,
                    step,
                } => {
                    let var_name = if let Some(web) = ssa.webs.iter().find(|w| w.original_reg == *var && !w.assigned_name.is_empty()) {
                        bump.alloc_str(&web.assigned_name)
                    } else {
                        bump.alloc_str(&format!("r{var}"))
                    };
                    let s = lower_expr(start, None, None, ssa, bump);
                    let l = lower_expr(limit, None, None, ssa, bump);
                    let st = step.as_ref().map(|sp| lower_expr(sp, None, None, ssa, bump));
                    out.push(crate::codegen::LuauStat::NumericFor {
                        var: var_name,
                        start: s,
                        limit: l,
                        step: st,
                        body: body_stats,
                    });
                }
                LoopKind::Generic { vars, iterators } => {
                    let var_names: Vec<&'arena str> = vars
                        .iter()
                        .map(|v| -> &'arena str {
                            if let Some(web) = ssa.webs.iter().find(|w| w.original_reg == *v && !w.assigned_name.is_empty()) {
                                &*bump.alloc_str(&web.assigned_name)
                            } else {
                                &*bump.alloc_str(&format!("r{v}"))
                            }
                        })
                        .collect();
                    let iters = iterators
                        .iter()
                        .map(|it| lower_expr(it, None, None, ssa, bump))
                        .collect();
                    out.push(crate::codegen::LuauStat::GenericFor {
                        vars: var_names,
                        iterators: iters,
                        body: body_stats,
                    });
                }
            }
        }
        StructuredRegion::Seq(regions) => {
            for r in regions {
                out.extend(lower_region(r, cfg, ssa, bump));
            }
        }
        StructuredRegion::Break => {
            out.push(crate::codegen::LuauStat::Break);
        }
        StructuredRegion::Continue => {
            out.push(crate::codegen::LuauStat::Continue);
        }
        StructuredRegion::Return(vals) => {
            let val_exprs = vals
                .iter()
                .map(|v| lower_expr(v, None, None, ssa, bump))
                .collect();
            out.push(crate::codegen::LuauStat::Return(val_exprs));
        }
        StructuredRegion::Crash => {
            let err_call = crate::codegen::LuauExpr::Call {
                func: Box::new(crate::codegen::LuauExpr::Identifier("error")),
                args: vec![crate::codegen::LuauExpr::String("LPH_CRASH")],
            };
            out.push(crate::codegen::LuauStat::Call(err_call));
        }
        StructuredRegion::Error(msg) => {
            let err_call = crate::codegen::LuauExpr::Call {
                func: Box::new(crate::codegen::LuauExpr::Identifier("error")),
                args: vec![crate::codegen::LuauExpr::String(bump.alloc_str(msg))],
            };
            out.push(crate::codegen::LuauStat::Call(err_call));
        }
    }
    out
}

/// Emits structured Luau code with SSA variable naming and Luau local budget limiter.
pub fn emit_luau_structured<'arena>(
    region: &StructuredRegion<'arena>,
    cfg: &ControlFlowGraph<'arena>,
    ssa: &crate::ssa::SsaContext,
    bump: &'arena bumpalo::Bump,
    outer_locals: usize,
    indent: usize,
) -> String {
    let stats = lower_region(region, cfg, ssa, bump);
    let fitted = crate::namer::scope::limit_locals(
        stats,
        outer_locals,
        crate::namer::scope::LUAU_LOCAL_BUDGET,
    );
    let transformed =
        crate::codegen::idioms::IdiomPipeline::default().transform(fitted, bump);
    let mut emitter =
        crate::codegen::emitter::LuauEmitter::new(crate::codegen::emitter::EmitterConfig::default());
    emitter.current_indent = indent;
    for st in &transformed {
        emitter.emit_stat(st);
    }
    emitter.buffer
}

