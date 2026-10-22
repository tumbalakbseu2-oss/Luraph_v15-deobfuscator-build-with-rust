//! Luau AST Idiom Transformation Engine.
//!
//! Provides five self-contained, semantics-preserving idiom passes:
//! 1. `rewrite_method_calls`: Recovers `obj:Method(...)` from `obj.Method(obj, ...)`.
//! 2. `rewrite_compound_assignments`: Recovers Luau compound assignments (`+=`, `-=`, etc.).
//! 3. `fold_table_constructors`: Folds sequential stores into table literal definitions.
//! 4. `coalesce_if_elseif`: Flattens nested `else\n if` ladders into `elseif` branches.
//! 5. `strip_trailing_returns`: Prunes redundant trailing `return` statements.

use std::collections::HashSet;

use bumpalo::Bump;

use crate::codegen::emitter::is_valid_ident;
use crate::codegen::{BinaryOp, ElseIfBranch, LuauExpr, LuauStat, LuauTableEntry, UnaryOp};

/// High-level idiom transformation pipeline orchestrator.
#[derive(Clone, Debug)]
pub struct IdiomPipeline {
    pub enable_method_calls: bool,
    pub enable_compound_assign: bool,
    pub enable_table_folding: bool,
    pub enable_if_coalescing: bool,
    pub enable_dead_tail_stripping: bool,
    pub enable_inlining: bool,
    pub enable_branch_simplification: bool,
}

impl Default for IdiomPipeline {
    fn default() -> Self {
        Self {
            enable_method_calls: true,
            enable_compound_assign: true,
            enable_table_folding: true,
            enable_if_coalescing: true,
            enable_dead_tail_stripping: true,
            enable_inlining: true,
            enable_branch_simplification: true,
        }
    }
}

impl IdiomPipeline {
    /// Executes all enabled idiom transformations over the statement sequence.
    pub fn transform<'arena>(
        &self,
        mut stats: Vec<LuauStat<'arena>>,
        bump: &'arena Bump,
    ) -> Vec<LuauStat<'arena>> {
        // Step 1: Initial inlining of single-def, single-use registers (e.g. table indexing, method lookup)
        if self.enable_inlining {
            stats = inline_temporaries(stats, bump);
        }
        // Step 2: Table constructor folding
        if self.enable_table_folding {
            stats = fold_table_constructors(stats, bump);
        }
        // Step 3: Compound assignments
        if self.enable_compound_assign {
            stats = rewrite_compound_assignments(stats);
        }
        // Step 4: Method call recovery (now sees direct table index calls)
        if self.enable_method_calls {
            stats = rewrite_method_calls(stats);
        }
        // Step 5: Second inlining pass (folds `obj` in `obj:Method(...)` if single-use)
        if self.enable_inlining {
            stats = inline_temporaries(stats, bump);
        }
        // Step 6: Empty branch simplification and dead branch pruning
        if self.enable_branch_simplification {
            stats = simplify_empty_branches(stats);
            stats = prune_dead_branches(stats);
        }
        // Step 7: Coalesce if-elseif ladders
        if self.enable_if_coalescing {
            stats = coalesce_if_elseif(stats);
        }
        // Step 8: Re-prune empty branches after coalescing
        if self.enable_branch_simplification {
            stats = prune_dead_branches(stats);
        }
        // Step 9: Strip trailing returns
        if self.enable_dead_tail_stripping {
            stats = strip_trailing_returns(stats);
        }
        // Step 10: Prune remaining dead local declarations (e.g. unused `local fn = fn_X`)
        if self.enable_inlining {
            stats = prune_dead_locals(stats);
        }
        stats
    }
}

// ============================================================================
// Pass 1: Method Call Sugar (`obj:Method(...)`)
// ============================================================================

/// Transforms `obj.Method(obj, ...)` or `obj["Method"](obj, ...)` to `obj:Method(...)`.
pub fn rewrite_method_calls<'arena>(stats: Vec<LuauStat<'arena>>) -> Vec<LuauStat<'arena>> {
    stats.into_iter().map(rewrite_stat_method_calls).collect()
}

fn rewrite_stat_method_calls<'arena>(stat: LuauStat<'arena>) -> LuauStat<'arena> {
    match stat {
        LuauStat::LocalAssign { names, values } => LuauStat::LocalAssign {
            names,
            values: values.into_iter().map(rewrite_expr_method_calls).collect(),
        },
        LuauStat::Assign { targets, values } => LuauStat::Assign {
            targets: targets.into_iter().map(rewrite_expr_method_calls).collect(),
            values: values.into_iter().map(rewrite_expr_method_calls).collect(),
        },
        LuauStat::CompoundAssign { target, op, value } => LuauStat::CompoundAssign {
            target: rewrite_expr_method_calls(target),
            op,
            value: rewrite_expr_method_calls(value),
        },
        LuauStat::Call(expr) => LuauStat::Call(rewrite_expr_method_calls(expr)),
        LuauStat::Return(vals) => {
            LuauStat::Return(vals.into_iter().map(rewrite_expr_method_calls).collect())
        }
        LuauStat::If {
            cond,
            then_branch,
            elseif_branches,
            else_branch,
        } => LuauStat::If {
            cond: rewrite_expr_method_calls(cond),
            then_branch: rewrite_method_calls(then_branch),
            elseif_branches: elseif_branches
                .into_iter()
                .map(|b| ElseIfBranch {
                    cond: rewrite_expr_method_calls(b.cond),
                    body: rewrite_method_calls(b.body),
                })
                .collect(),
            else_branch: else_branch.map(rewrite_method_calls),
        },
        LuauStat::While { cond, body } => LuauStat::While {
            cond: rewrite_expr_method_calls(cond),
            body: rewrite_method_calls(body),
        },
        LuauStat::Repeat { cond, body } => LuauStat::Repeat {
            cond: rewrite_expr_method_calls(cond),
            body: rewrite_method_calls(body),
        },
        LuauStat::NumericFor {
            var,
            start,
            limit,
            step,
            body,
        } => LuauStat::NumericFor {
            var,
            start: rewrite_expr_method_calls(start),
            limit: rewrite_expr_method_calls(limit),
            step: step.map(rewrite_expr_method_calls),
            body: rewrite_method_calls(body),
        },
        LuauStat::GenericFor {
            vars,
            iterators,
            body,
        } => LuauStat::GenericFor {
            vars,
            iterators: iterators
                .into_iter()
                .map(rewrite_expr_method_calls)
                .collect(),
            body: rewrite_method_calls(body),
        },
        LuauStat::LocalFunctionDef {
            name,
            params,
            is_vararg,
            body,
        } => LuauStat::LocalFunctionDef {
            name,
            params,
            is_vararg,
            body: rewrite_method_calls(body),
        },
        LuauStat::FunctionDef {
            name,
            params,
            is_vararg,
            body,
        } => LuauStat::FunctionDef {
            name,
            params,
            is_vararg,
            body: rewrite_method_calls(body),
        },
        LuauStat::DoBlock(body) => LuauStat::DoBlock(rewrite_method_calls(body)),
        other => other,
    }
}

fn rewrite_expr_method_calls<'arena>(expr: LuauExpr<'arena>) -> LuauExpr<'arena> {
    match expr {
        LuauExpr::Call { func, args } => {
            let func = rewrite_expr_method_calls(*func);
            let mut args: Vec<LuauExpr<'arena>> =
                args.into_iter().map(rewrite_expr_method_calls).collect();

            if let LuauExpr::Index { table, key } = func {
                let method_name = match key.as_ref() {
                    LuauExpr::String(s) if is_valid_ident(s) => Some(*s),
                    LuauExpr::Identifier(s) if is_valid_ident(s) => Some(*s),
                    _ => None,
                };

                if let Some(method) = method_name {
                    if !args.is_empty()
                        && expressions_match(&table, &args[0])
                        && is_pure_expr(&table)
                    {
                        args.remove(0);
                        return LuauExpr::MethodCall {
                            target: table,
                            method,
                            args,
                        };
                    }
                }

                LuauExpr::Call {
                    func: Box::new(LuauExpr::Index { table, key }),
                    args,
                }
            } else {
                LuauExpr::Call {
                    func: Box::new(func),
                    args,
                }
            }
        }
        LuauExpr::MethodCall {
            target,
            method,
            args,
        } => LuauExpr::MethodCall {
            target: Box::new(rewrite_expr_method_calls(*target)),
            method,
            args: args.into_iter().map(rewrite_expr_method_calls).collect(),
        },
        LuauExpr::Unary { op, expr } => LuauExpr::Unary {
            op,
            expr: Box::new(rewrite_expr_method_calls(*expr)),
        },
        LuauExpr::Binary { op, left, right } => LuauExpr::Binary {
            op,
            left: Box::new(rewrite_expr_method_calls(*left)),
            right: Box::new(rewrite_expr_method_calls(*right)),
        },
        LuauExpr::Index { table, key } => LuauExpr::Index {
            table: Box::new(rewrite_expr_method_calls(*table)),
            key: Box::new(rewrite_expr_method_calls(*key)),
        },
        LuauExpr::Table(entries) => LuauExpr::Table(
            entries
                .into_iter()
                .map(|e| match e {
                    LuauTableEntry::Array(v) => LuauTableEntry::Array(rewrite_expr_method_calls(v)),
                    LuauTableEntry::Value(v) => LuauTableEntry::Value(rewrite_expr_method_calls(v)),
                    LuauTableEntry::Record(k, v) => {
                        LuauTableEntry::Record(k, rewrite_expr_method_calls(v))
                    }
                    LuauTableEntry::General(k, v) => LuauTableEntry::General(
                        rewrite_expr_method_calls(k),
                        rewrite_expr_method_calls(v),
                    ),
                })
                .collect(),
        ),
        LuauExpr::IfExp {
            cond,
            then_expr,
            else_expr,
        } => LuauExpr::IfExp {
            cond: Box::new(rewrite_expr_method_calls(*cond)),
            then_expr: Box::new(rewrite_expr_method_calls(*then_expr)),
            else_expr: Box::new(rewrite_expr_method_calls(*else_expr)),
        },
        LuauExpr::Function {
            params,
            is_vararg,
            body,
        } => LuauExpr::Function {
            params,
            is_vararg,
            body: rewrite_method_calls(body),
        },
        other => other,
    }
}

// ============================================================================
// Pass 2: Compound Assignments (`target += value`)
// ============================================================================

/// Transforms `x = x + v` into Luau compound assignments (`+=`, `-=`, `*=`, `/=`, `^=`, `%=`, `..=`).
pub fn rewrite_compound_assignments<'arena>(
    stats: Vec<LuauStat<'arena>>,
) -> Vec<LuauStat<'arena>> {
    stats
        .into_iter()
        .map(rewrite_stat_compound_assignment)
        .collect()
}

fn rewrite_stat_compound_assignment<'arena>(stat: LuauStat<'arena>) -> LuauStat<'arena> {
    match stat {
        LuauStat::Assign {
            mut targets,
            mut values,
        } => {
            if targets.len() == 1 && values.len() == 1 {
                let target = targets.remove(0);
                let value = values.remove(0);

                if let LuauExpr::Binary { op, left, right } = value {
                    if expressions_match(&target, &left) && is_pure_lvalue(&target) {
                        if let Some(compound_op) = op.to_compound() {
                            return LuauStat::CompoundAssign {
                                target,
                                op: compound_op,
                                value: *right,
                            };
                        }
                    }
                    LuauStat::Assign {
                        targets: vec![target],
                        values: vec![LuauExpr::Binary { op, left, right }],
                    }
                } else {
                    LuauStat::Assign {
                        targets: vec![target],
                        values: vec![value],
                    }
                }
            } else {
                LuauStat::Assign { targets, values }
            }
        }
        LuauStat::If {
            cond,
            then_branch,
            elseif_branches,
            else_branch,
        } => LuauStat::If {
            cond,
            then_branch: rewrite_compound_assignments(then_branch),
            elseif_branches: elseif_branches
                .into_iter()
                .map(|b| ElseIfBranch {
                    cond: b.cond,
                    body: rewrite_compound_assignments(b.body),
                })
                .collect(),
            else_branch: else_branch.map(rewrite_compound_assignments),
        },
        LuauStat::While { cond, body } => LuauStat::While {
            cond,
            body: rewrite_compound_assignments(body),
        },
        LuauStat::Repeat { cond, body } => LuauStat::Repeat {
            cond,
            body: rewrite_compound_assignments(body),
        },
        LuauStat::NumericFor {
            var,
            start,
            limit,
            step,
            body,
        } => LuauStat::NumericFor {
            var,
            start,
            limit,
            step,
            body: rewrite_compound_assignments(body),
        },
        LuauStat::GenericFor {
            vars,
            iterators,
            body,
        } => LuauStat::GenericFor {
            vars,
            iterators,
            body: rewrite_compound_assignments(body),
        },
        LuauStat::LocalFunctionDef {
            name,
            params,
            is_vararg,
            body,
        } => LuauStat::LocalFunctionDef {
            name,
            params,
            is_vararg,
            body: rewrite_compound_assignments(body),
        },
        LuauStat::FunctionDef {
            name,
            params,
            is_vararg,
            body,
        } => LuauStat::FunctionDef {
            name,
            params,
            is_vararg,
            body: rewrite_compound_assignments(body),
        },
        LuauStat::DoBlock(body) => LuauStat::DoBlock(rewrite_compound_assignments(body)),
        other => other,
    }
}

// ============================================================================
// Pass 3: Table Constructor Folding
// ============================================================================

/// Folds sequential table stores into table constructor definitions `{ ... }`.
pub fn fold_table_constructors<'arena>(
    stats: Vec<LuauStat<'arena>>,
    bump: &'arena Bump,
) -> Vec<LuauStat<'arena>> {
    let mut out = Vec::with_capacity(stats.len());
    let mut i = 0;

    while i < stats.len() {
        let current = &stats[i];

        // Check for `local t = { ... }`
        if let LuauStat::LocalAssign { names, values } = current {
            if names.len() == 1 && values.len() == 1 {
                let target_name = names[0];
                if let LuauExpr::Table(existing_entries) = &values[0] {
                    let mut folded_entries = existing_entries.clone();
                    let mut next_array_idx = 1.0;

                    for e in &folded_entries {
                        if matches!(e, LuauTableEntry::Array(_) | LuauTableEntry::Value(_)) {
                            next_array_idx += 1.0;
                        }
                    }

                    let mut j = i + 1;
                    let mut consumed_indices = HashSet::new();

                    while j < stats.len() {
                        let candidate = &stats[j];

                        if let LuauStat::Assign { targets, values: assign_vals } = candidate {
                            if targets.len() == 1 && assign_vals.len() == 1 {
                                let tgt = &targets[0];
                                let val = &assign_vals[0];

                                if let LuauExpr::Index { table, key } = tgt {
                                    if let LuauExpr::Identifier(tbl_id) = table.as_ref() {
                                        if *tbl_id == target_name
                                            && !expr_references_ident(val, target_name)
                                        {
                                            match key.as_ref() {
                                                LuauExpr::Number(n)
                                                    if (*n - next_array_idx).abs()
                                                        < f64::EPSILON =>
                                                {
                                                    folded_entries
                                                        .push(LuauTableEntry::Array(val.clone()));
                                                    next_array_idx += 1.0;
                                                    consumed_indices.insert(j);
                                                    j += 1;
                                                    continue;
                                                }
                                                LuauExpr::String(prop)
                                                    if is_valid_ident(prop) =>
                                                {
                                                    folded_entries.push(LuauTableEntry::Record(
                                                        prop,
                                                        val.clone(),
                                                    ));
                                                    consumed_indices.insert(j);
                                                    j += 1;
                                                    continue;
                                                }
                                                _ => {
                                                    folded_entries.push(LuauTableEntry::General(
                                                        *key.clone(),
                                                        val.clone(),
                                                    ));
                                                    consumed_indices.insert(j);
                                                    j += 1;
                                                    continue;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // If statement references target_name, stop folding to preserve ordering
                        if stat_references_ident(candidate, target_name) {
                            break;
                        }

                        j += 1;
                    }

                    if !consumed_indices.is_empty() {
                        out.push(LuauStat::LocalAssign {
                            names: vec![target_name],
                            values: vec![LuauExpr::Table(folded_entries)],
                        });

                        for k in (i + 1)..j {
                            if !consumed_indices.contains(&k) {
                                out.push(fold_stat_constructors(stats[k].clone(), bump));
                            }
                        }
                        i = j;
                        continue;
                    }
                }
            }
        }

        out.push(fold_stat_constructors(stats[i].clone(), bump));
        i += 1;
    }

    out
}

fn fold_stat_constructors<'arena>(
    stat: LuauStat<'arena>,
    bump: &'arena Bump,
) -> LuauStat<'arena> {
    match stat {
        LuauStat::If {
            cond,
            then_branch,
            elseif_branches,
            else_branch,
        } => LuauStat::If {
            cond,
            then_branch: fold_table_constructors(then_branch, bump),
            elseif_branches: elseif_branches
                .into_iter()
                .map(|b| ElseIfBranch {
                    cond: b.cond,
                    body: fold_table_constructors(b.body, bump),
                })
                .collect(),
            else_branch: else_branch.map(|b| fold_table_constructors(b, bump)),
        },
        LuauStat::While { cond, body } => LuauStat::While {
            cond,
            body: fold_table_constructors(body, bump),
        },
        LuauStat::Repeat { cond, body } => LuauStat::Repeat {
            cond,
            body: fold_table_constructors(body, bump),
        },
        LuauStat::NumericFor {
            var,
            start,
            limit,
            step,
            body,
        } => LuauStat::NumericFor {
            var,
            start,
            limit,
            step,
            body: fold_table_constructors(body, bump),
        },
        LuauStat::GenericFor {
            vars,
            iterators,
            body,
        } => LuauStat::GenericFor {
            vars,
            iterators,
            body: fold_table_constructors(body, bump),
        },
        LuauStat::LocalFunctionDef {
            name,
            params,
            is_vararg,
            body,
        } => LuauStat::LocalFunctionDef {
            name,
            params,
            is_vararg,
            body: fold_table_constructors(body, bump),
        },
        LuauStat::FunctionDef {
            name,
            params,
            is_vararg,
            body,
        } => LuauStat::FunctionDef {
            name,
            params,
            is_vararg,
            body: fold_table_constructors(body, bump),
        },
        LuauStat::DoBlock(body) => LuauStat::DoBlock(fold_table_constructors(body, bump)),
        other => other,
    }
}

// ============================================================================
// Pass 4: Coalesce If-ElseIf Ladders
// ============================================================================

/// Collapses nested `else\n if` blocks into clean `elseif` ladders.
pub fn coalesce_if_elseif<'arena>(stats: Vec<LuauStat<'arena>>) -> Vec<LuauStat<'arena>> {
    stats.into_iter().map(coalesce_stat_if_elseif).collect()
}

fn coalesce_stat_if_elseif<'arena>(stat: LuauStat<'arena>) -> LuauStat<'arena> {
    match stat {
        LuauStat::If {
            cond,
            then_branch,
            mut elseif_branches,
            else_branch,
        } => {
            let then_branch = coalesce_if_elseif(then_branch);
            elseif_branches = elseif_branches
                .into_iter()
                .map(|b| ElseIfBranch {
                    cond: b.cond,
                    body: coalesce_if_elseif(b.body),
                })
                .collect();
            let mut else_branch = else_branch.map(coalesce_if_elseif);

            while let Some(mut els) = else_branch.take() {
                if els.len() == 1 && matches!(els.first(), Some(LuauStat::If { .. })) {
                    if let Some(LuauStat::If {
                        cond: c2,
                        then_branch: t2,
                        elseif_branches: e2,
                        else_branch: el2,
                    }) = els.pop()
                    {
                        elseif_branches.push(ElseIfBranch {
                            cond: c2,
                            body: t2,
                        });
                        elseif_branches.extend(e2);
                        else_branch = el2;
                        continue;
                    }
                }
                else_branch = Some(els);
                break;
            }

            LuauStat::If {
                cond,
                then_branch,
                elseif_branches,
                else_branch,
            }
        }
        LuauStat::While { cond, body } => LuauStat::While {
            cond,
            body: coalesce_if_elseif(body),
        },
        LuauStat::Repeat { cond, body } => LuauStat::Repeat {
            cond,
            body: coalesce_if_elseif(body),
        },
        LuauStat::NumericFor {
            var,
            start,
            limit,
            step,
            body,
        } => LuauStat::NumericFor {
            var,
            start,
            limit,
            step,
            body: coalesce_if_elseif(body),
        },
        LuauStat::GenericFor {
            vars,
            iterators,
            body,
        } => LuauStat::GenericFor {
            vars,
            iterators,
            body: coalesce_if_elseif(body),
        },
        LuauStat::LocalFunctionDef {
            name,
            params,
            is_vararg,
            body,
        } => LuauStat::LocalFunctionDef {
            name,
            params,
            is_vararg,
            body: coalesce_if_elseif(body),
        },
        LuauStat::FunctionDef {
            name,
            params,
            is_vararg,
            body,
        } => LuauStat::FunctionDef {
            name,
            params,
            is_vararg,
            body: coalesce_if_elseif(body),
        },
        LuauStat::DoBlock(body) => LuauStat::DoBlock(coalesce_if_elseif(body)),
        other => other,
    }
}

// ============================================================================
// Pass 5: Strip Redundant Trailing Returns
// ============================================================================

/// Strips redundant bare `return` statements from the end of blocks and functions.
pub fn strip_trailing_returns<'arena>(mut stats: Vec<LuauStat<'arena>>) -> Vec<LuauStat<'arena>> {
    if let Some(LuauStat::Return(vals)) = stats.last() {
        if vals.is_empty() {
            stats.pop();
        }
    }

    stats
        .into_iter()
        .map(|st| match st {
            LuauStat::If {
                cond,
                then_branch,
                elseif_branches,
                else_branch,
            } => LuauStat::If {
                cond,
                then_branch: strip_trailing_returns(then_branch),
                elseif_branches: elseif_branches
                    .into_iter()
                    .map(|b| ElseIfBranch {
                        cond: b.cond,
                        body: strip_trailing_returns(b.body),
                    })
                    .collect(),
                else_branch: else_branch.map(strip_trailing_returns),
            },
            LuauStat::While { cond, body } => LuauStat::While {
                cond,
                body: strip_trailing_returns(body),
            },
            LuauStat::Repeat { cond, body } => LuauStat::Repeat {
                cond,
                body: strip_trailing_returns(body),
            },
            LuauStat::NumericFor {
                var,
                start,
                limit,
                step,
                body,
            } => LuauStat::NumericFor {
                var,
                start,
                limit,
                step,
                body: strip_trailing_returns(body),
            },
            LuauStat::GenericFor {
                vars,
                iterators,
                body,
            } => LuauStat::GenericFor {
                vars,
                iterators,
                body: strip_trailing_returns(body),
            },
            LuauStat::LocalFunctionDef {
                name,
                params,
                is_vararg,
                body,
            } => LuauStat::LocalFunctionDef {
                name,
                params,
                is_vararg,
                body: strip_trailing_returns(body),
            },
            LuauStat::FunctionDef {
                name,
                params,
                is_vararg,
                body,
            } => LuauStat::FunctionDef {
                name,
                params,
                is_vararg,
                body: strip_trailing_returns(body),
            },
            LuauStat::DoBlock(body) => LuauStat::DoBlock(strip_trailing_returns(body)),
            other => other,
        })
        .collect()
}

// ============================================================================
// Expression & Statement Analysis Helpers
// ============================================================================

/// Structural expression equivalence checking.
pub fn expressions_match(a: &LuauExpr, b: &LuauExpr) -> bool {
    match (a, b) {
        (LuauExpr::Identifier(id1), LuauExpr::Identifier(id2)) => id1 == id2,
        (LuauExpr::String(s1), LuauExpr::String(s2)) => s1 == s2,
        (LuauExpr::Number(n1), LuauExpr::Number(n2)) => {
            (n1 - n2).abs() < f64::EPSILON || (n1.is_nan() && n2.is_nan())
        }
        (LuauExpr::Boolean(b1), LuauExpr::Boolean(b2)) => b1 == b2,
        (LuauExpr::Nil, LuauExpr::Nil) => true,
        (LuauExpr::Vararg, LuauExpr::Vararg) => true,
        (
            LuauExpr::Index {
                table: t1,
                key: k1,
            },
            LuauExpr::Index {
                table: t2,
                key: k2,
            },
        ) => expressions_match(t1, t2) && expressions_match(k1, k2),
        (
            LuauExpr::Binary {
                op: op1,
                left: l1,
                right: r1,
            },
            LuauExpr::Binary {
                op: op2,
                left: l2,
                right: r2,
            },
        ) => op1 == op2 && expressions_match(l1, l2) && expressions_match(r1, r2),
        (LuauExpr::Unary { op: op1, expr: e1 }, LuauExpr::Unary { op: op2, expr: e2 }) => {
            op1 == op2 && expressions_match(e1, e2)
        }
        _ => false,
    }
}

/// Checks whether an expression is side-effect-free (pure).
pub fn is_pure_expr(expr: &LuauExpr) -> bool {
    match expr {
        LuauExpr::Nil
        | LuauExpr::Boolean(_)
        | LuauExpr::Number(_)
        | LuauExpr::String(_)
        | LuauExpr::Identifier(_)
        | LuauExpr::Vararg => true,
        LuauExpr::Index { table, key } => is_pure_expr(table) && is_pure_expr(key),
        LuauExpr::Unary { expr, .. } => is_pure_expr(expr),
        LuauExpr::Binary { left, right, .. } => is_pure_expr(left) && is_pure_expr(right),
        _ => false,
    }
}

/// Checks whether an expression is a valid, pure assignment target (lvalue).
pub fn is_pure_lvalue(expr: &LuauExpr) -> bool {
    match expr {
        LuauExpr::Identifier(_) => true,
        LuauExpr::Index { table, key } => is_pure_expr(table) && is_pure_expr(key),
        _ => false,
    }
}

/// Checks if an expression references a specific variable identifier.
pub fn expr_references_ident(expr: &LuauExpr, ident: &str) -> bool {
    match expr {
        LuauExpr::Identifier(id) => *id == ident,
        LuauExpr::Index { table, key } => {
            expr_references_ident(table, ident) || expr_references_ident(key, ident)
        }
        LuauExpr::Call { func, args } => {
            expr_references_ident(func, ident)
                || args.iter().any(|a| expr_references_ident(a, ident))
        }
        LuauExpr::MethodCall { target, args, .. } => {
            expr_references_ident(target, ident)
                || args.iter().any(|a| expr_references_ident(a, ident))
        }
        LuauExpr::Unary { expr, .. } => expr_references_ident(expr, ident),
        LuauExpr::Binary { left, right, .. } => {
            expr_references_ident(left, ident) || expr_references_ident(right, ident)
        }
        LuauExpr::IfExp {
            cond,
            then_expr,
            else_expr,
        } => {
            expr_references_ident(cond, ident)
                || expr_references_ident(then_expr, ident)
                || expr_references_ident(else_expr, ident)
        }
        LuauExpr::Table(entries) => entries.iter().any(|e| match e {
            LuauTableEntry::Array(v) | LuauTableEntry::Value(v) => {
                expr_references_ident(v, ident)
            }
            LuauTableEntry::Record(_, v) => expr_references_ident(v, ident),
            LuauTableEntry::General(k, v) => {
                expr_references_ident(k, ident) || expr_references_ident(v, ident)
            }
        }),
        LuauExpr::Function { params, body, .. } => {
            if params.contains(&ident) {
                false
            } else {
                body.iter().any(|s| stat_references_ident(s, ident))
            }
        }
        _ => false,
    }
}

/// Checks if a statement references a specific variable identifier.
pub fn stat_references_ident(stat: &LuauStat, ident: &str) -> bool {
    match stat {
        LuauStat::LocalDecl(names) => names.contains(&ident),
        LuauStat::LocalAssign { names, values } => {
            names.contains(&ident) || values.iter().any(|v| expr_references_ident(v, ident))
        }
        LuauStat::Assign { targets, values } => {
            targets.iter().any(|t| expr_references_ident(t, ident))
                || values.iter().any(|v| expr_references_ident(v, ident))
        }
        LuauStat::CompoundAssign { target, value, .. } => {
            expr_references_ident(target, ident) || expr_references_ident(value, ident)
        }
        LuauStat::Call(expr) => expr_references_ident(expr, ident),
        LuauStat::Return(vals) => vals.iter().any(|v| expr_references_ident(v, ident)),
        LuauStat::If {
            cond,
            then_branch,
            elseif_branches,
            else_branch,
        } => {
            expr_references_ident(cond, ident)
                || then_branch.iter().any(|s| stat_references_ident(s, ident))
                || elseif_branches.iter().any(|b| {
                    expr_references_ident(&b.cond, ident)
                        || b.body.iter().any(|s| stat_references_ident(s, ident))
                })
                || else_branch
                    .as_ref()
                    .map_or(false, |els| els.iter().any(|s| stat_references_ident(s, ident)))
        }
        LuauStat::While { cond, body } | LuauStat::Repeat { cond, body } => {
            expr_references_ident(cond, ident)
                || body.iter().any(|s| stat_references_ident(s, ident))
        }
        LuauStat::NumericFor {
            var,
            start,
            limit,
            step,
            body,
        } => {
            if *var == ident {
                false
            } else {
                expr_references_ident(start, ident)
                    || expr_references_ident(limit, ident)
                    || step.as_ref().map_or(false, |s| expr_references_ident(s, ident))
                    || body.iter().any(|s| stat_references_ident(s, ident))
            }
        }
        LuauStat::GenericFor {
            vars,
            iterators,
            body,
        } => {
            if vars.contains(&ident) {
                false
            } else {
                iterators.iter().any(|it| expr_references_ident(it, ident))
                    || body.iter().any(|s| stat_references_ident(s, ident))
            }
        }
        LuauStat::LocalFunctionDef {
            name,
            params,
            body,
            ..
        } => {
            *name == ident
                || (!params.contains(&ident)
                    && body.iter().any(|s| stat_references_ident(s, ident)))
        }
        LuauStat::FunctionDef {
            name,
            params,
            body,
            ..
        } => {
            *name == ident
                || (!params.contains(&ident)
                    && body.iter().any(|s| stat_references_ident(s, ident)))
        }
        LuauStat::DoBlock(body) => body.iter().any(|s| stat_references_ident(s, ident)),
        _ => false,
    }
}

// ============================================================================
// Pass 6: Temporary Variable Inlining & Copy Propagation
// ============================================================================

/// Folds single-definition, single-use local variables into their point of use,
/// strictly preserving Luau evaluation ordering and side-effect semantics.
pub fn inline_temporaries<'arena>(
    mut stats: Vec<LuauStat<'arena>>,
    bump: &'arena Bump,
) -> Vec<LuauStat<'arena>> {
    let mut changed = true;
    let mut iteration = 0;
    const MAX_ITERATIONS: usize = 8;

    while changed && iteration < MAX_ITERATIONS {
        changed = false;
        iteration += 1;

        let mut out = Vec::with_capacity(stats.len());
        let mut i = 0;

        while i < stats.len() {
            let single_def = if let LuauStat::LocalAssign { names, values } = &stats[i] {
                if names.len() == 1 && values.len() == 1 {
                    Some((names[0], values[0].clone()))
                } else {
                    None
                }
            } else {
                None
            };

            if let Some((target_name, val_expr)) = single_def {
                // Skip self-assignments like local a = a
                if let LuauExpr::Identifier(src_id) = &val_expr {
                    if *src_id == target_name {
                        out.push(stats[i].clone());
                        i += 1;
                        continue;
                    }
                }

                // Find all uses of target_name in subsequent statements
                let mut use_indices = Vec::new();
                for (k, future_stat) in stats.iter().enumerate().skip(i + 1) {
                    let count = count_stat_ident(future_stat, target_name);
                    if count > 0 {
                        use_indices.push((k, count));
                    }
                }

                // Case A: Single use in subsequent code
                if use_indices.len() == 1 && use_indices[0].1 == 1 {
                    let (use_idx, _) = use_indices[0];
                    let intervening = &stats[i + 1..use_idx];

                    if can_safely_inline(&val_expr, &stats[use_idx], intervening, target_name) {
                        substitute_stat_ident(&mut stats[use_idx], target_name, &val_expr);
                        changed = true;
                        i += 1;
                        continue;
                    }
                }

                // Case B: Copy propagation (pure identifier alias or literal constant across multiple uses)
                if is_pure_copy_expr(&val_expr) && !use_indices.is_empty() {
                    let intervening_clobbered = stats.iter().skip(i + 1).any(|st| {
                        if let LuauExpr::Identifier(src_id) = &val_expr {
                            stat_mutates_ident(st, src_id)
                        } else {
                            false
                        }
                    });

                    if !intervening_clobbered {
                        for (use_idx, _) in &use_indices {
                            substitute_stat_ident(&mut stats[*use_idx], target_name, &val_expr);
                        }
                        changed = true;
                        i += 1;
                        continue;
                    }
                }
            }

            out.push(stats[i].clone());
            i += 1;
        }

        stats = out;
    }

    // Recursively process nested block structures
    stats
        .into_iter()
        .map(|st| match st {
            LuauStat::If {
                cond,
                then_branch,
                elseif_branches,
                else_branch,
            } => LuauStat::If {
                cond,
                then_branch: inline_temporaries(then_branch, bump),
                elseif_branches: elseif_branches
                    .into_iter()
                    .map(|b| ElseIfBranch {
                        cond: b.cond,
                        body: inline_temporaries(b.body, bump),
                    })
                    .collect(),
                else_branch: else_branch.map(|b| inline_temporaries(b, bump)),
            },
            LuauStat::While { cond, body } => LuauStat::While {
                cond,
                body: inline_temporaries(body, bump),
            },
            LuauStat::Repeat { cond, body } => LuauStat::Repeat {
                cond,
                body: inline_temporaries(body, bump),
            },
            LuauStat::NumericFor {
                var,
                start,
                limit,
                step,
                body,
            } => LuauStat::NumericFor {
                var,
                start,
                limit,
                step,
                body: inline_temporaries(body, bump),
            },
            LuauStat::GenericFor {
                vars,
                iterators,
                body,
            } => LuauStat::GenericFor {
                vars,
                iterators,
                body: inline_temporaries(body, bump),
            },
            LuauStat::LocalFunctionDef {
                name,
                params,
                is_vararg,
                body,
            } => LuauStat::LocalFunctionDef {
                name,
                params,
                is_vararg,
                body: inline_temporaries(body, bump),
            },
            LuauStat::FunctionDef {
                name,
                params,
                is_vararg,
                body,
            } => LuauStat::FunctionDef {
                name,
                params,
                is_vararg,
                body: inline_temporaries(body, bump),
            },
            LuauStat::DoBlock(body) => LuauStat::DoBlock(inline_temporaries(body, bump)),
            other => other,
        })
        .collect()
}

/// Prunes dead local variable declarations that have zero references in subsequent statements.
pub fn prune_dead_locals<'arena>(stats: Vec<LuauStat<'arena>>) -> Vec<LuauStat<'arena>> {
    let mut out = Vec::with_capacity(stats.len());

    for (i, st) in stats.iter().enumerate() {
        if let LuauStat::LocalAssign { names, values } = st {
            if names.len() == 1 && values.len() == 1 {
                let name = names[0];
                let is_referenced = stats.iter().skip(i + 1).any(|s| stat_references_ident(s, name));

                if !is_referenced {
                    let val = &values[0];
                    if is_pure_expr(val) || matches!(val, LuauExpr::Identifier(_)) {
                        // Completely dead pure assignment -> eliminate
                        continue;
                    } else if matches!(val, LuauExpr::Call { .. } | LuauExpr::MethodCall { .. }) {
                        // Preserve side effect of function call, discard unused return
                        out.push(LuauStat::Call(val.clone()));
                        continue;
                    }
                }
            }
        }
        out.push(st.clone());
    }

    out.into_iter()
        .map(|st| match st {
            LuauStat::If {
                cond,
                then_branch,
                elseif_branches,
                else_branch,
            } => LuauStat::If {
                cond,
                then_branch: prune_dead_locals(then_branch),
                elseif_branches: elseif_branches
                    .into_iter()
                    .map(|b| ElseIfBranch {
                        cond: b.cond,
                        body: prune_dead_locals(b.body),
                    })
                    .collect(),
                else_branch: else_branch.map(prune_dead_locals),
            },
            LuauStat::While { cond, body } => LuauStat::While {
                cond,
                body: prune_dead_locals(body),
            },
            LuauStat::Repeat { cond, body } => LuauStat::Repeat {
                cond,
                body: prune_dead_locals(body),
            },
            LuauStat::NumericFor {
                var,
                start,
                limit,
                step,
                body,
            } => LuauStat::NumericFor {
                var,
                start,
                limit,
                step,
                body: prune_dead_locals(body),
            },
            LuauStat::GenericFor {
                vars,
                iterators,
                body,
            } => LuauStat::GenericFor {
                vars,
                iterators,
                body: prune_dead_locals(body),
            },
            LuauStat::LocalFunctionDef {
                name,
                params,
                is_vararg,
                body,
            } => LuauStat::LocalFunctionDef {
                name,
                params,
                is_vararg,
                body: prune_dead_locals(body),
            },
            LuauStat::FunctionDef {
                name,
                params,
                is_vararg,
                body,
            } => LuauStat::FunctionDef {
                name,
                params,
                is_vararg,
                body: prune_dead_locals(body),
            },
            LuauStat::DoBlock(body) => LuauStat::DoBlock(prune_dead_locals(body)),
            other => other,
        })
        .collect()
}

// ============================================================================
// Expression Inlining Analysis Helpers
// ============================================================================

/// Checks whether an expression can safely be inlined into a target statement across intervening statements.
fn can_safely_inline(
    expr: &LuauExpr,
    target_stat: &LuauStat,
    intervening: &[LuauStat],
    target_name: &str,
) -> bool {
    // Rule 0: Cannot inline if target statement mutates target_name (e.g. assignment target)
    if stat_mutates_ident(target_stat, target_name) {
        return false;
    }

    let is_pure = is_pure_expr(expr);

    // Rule 1: Impure expressions (e.g. calls) cannot move across any intervening statements
    if !is_pure && !intervening.is_empty() {
        return false;
    }

    // Rule 2: Impure expressions cannot be inlined into loop bodies (would change execution frequency)
    if !is_pure && is_loop_stat(target_stat) {
        return false;
    }

    // Rule 3: Intervening statements must not mutate any variables read by expr
    let mut read_vars = HashSet::new();
    collect_expr_identifiers(expr, &mut read_vars);
    for st in intervening {
        for var in &read_vars {
            if stat_mutates_ident(st, var) {
                return false;
            }
        }
    }

    // Rule 4: If expr reads memory (Index or Call), intervening statements must not have side effects
    if expr_reads_memory(expr) {
        for st in intervening {
            if stat_has_side_effects(st) {
                return false;
            }
        }
    }

    // Rule 5: Luau Multret Protection: Calls cannot be inlined into the last argument of a call or return
    if matches!(expr, LuauExpr::Call { .. } | LuauExpr::MethodCall { .. }) {
        if is_tail_multret_position(target_stat, target_name) {
            return false;
        }
    }

    true
}

fn is_pure_copy_expr(expr: &LuauExpr) -> bool {
    matches!(
        expr,
        LuauExpr::Nil
            | LuauExpr::Boolean(_)
            | LuauExpr::Number(_)
            | LuauExpr::String(_)
            | LuauExpr::Identifier(_)
    )
}

fn is_loop_stat(stat: &LuauStat) -> bool {
    matches!(
        stat,
        LuauStat::While { .. }
            | LuauStat::Repeat { .. }
            | LuauStat::NumericFor { .. }
            | LuauStat::GenericFor { .. }
    )
}

fn expr_reads_memory(expr: &LuauExpr) -> bool {
    match expr {
        LuauExpr::Index { .. } | LuauExpr::Call { .. } | LuauExpr::MethodCall { .. } => true,
        LuauExpr::Unary { expr, .. } => expr_reads_memory(expr),
        LuauExpr::Binary { left, right, .. } => expr_reads_memory(left) || expr_reads_memory(right),
        _ => false,
    }
}

fn stat_has_side_effects(stat: &LuauStat) -> bool {
    match stat {
        LuauStat::Call(_)
        | LuauStat::Assign { .. }
        | LuauStat::CompoundAssign { .. }
        | LuauStat::Return(_) => true,
        LuauStat::If { then_branch, elseif_branches, else_branch, .. } => {
            then_branch.iter().any(stat_has_side_effects)
                || elseif_branches.iter().any(|b| b.body.iter().any(stat_has_side_effects))
                || else_branch.as_ref().map_or(false, |b| b.iter().any(stat_has_side_effects))
        }
        LuauStat::While { body, .. }
        | LuauStat::Repeat { body, .. }
        | LuauStat::NumericFor { body, .. }
        | LuauStat::GenericFor { body, .. }
        | LuauStat::DoBlock(body) => body.iter().any(stat_has_side_effects),
        _ => false,
    }
}

fn stat_mutates_ident(stat: &LuauStat, ident: &str) -> bool {
    match stat {
        LuauStat::Assign { targets, .. } => targets.iter().any(|t| {
            if let LuauExpr::Identifier(id) = t {
                *id == ident
            } else {
                false
            }
        }),
        LuauStat::CompoundAssign { target, .. } => {
            if let LuauExpr::Identifier(id) = target {
                *id == ident
            } else {
                false
            }
        }
        LuauStat::LocalAssign { names, .. } | LuauStat::LocalDecl(names) => names.contains(&ident),
        _ => false,
    }
}

fn collect_expr_identifiers<'arena>(expr: &LuauExpr<'arena>, out: &mut HashSet<&'arena str>) {
    match expr {
        LuauExpr::Identifier(id) => {
            out.insert(*id);
        }
        LuauExpr::Unary { expr, .. } => collect_expr_identifiers(expr, out),
        LuauExpr::Binary { left, right, .. } => {
            collect_expr_identifiers(left, out);
            collect_expr_identifiers(right, out);
        }
        LuauExpr::Index { table, key } => {
            collect_expr_identifiers(table, out);
            collect_expr_identifiers(key, out);
        }
        LuauExpr::Call { func, args } => {
            collect_expr_identifiers(func, out);
            for a in args {
                collect_expr_identifiers(a, out);
            }
        }
        LuauExpr::MethodCall { target, args, .. } => {
            collect_expr_identifiers(target, out);
            for a in args {
                collect_expr_identifiers(a, out);
            }
        }
        LuauExpr::IfExp { cond, then_expr, else_expr } => {
            collect_expr_identifiers(cond, out);
            collect_expr_identifiers(then_expr, out);
            collect_expr_identifiers(else_expr, out);
        }
        LuauExpr::Table(entries) => {
            for e in entries {
                match e {
                    LuauTableEntry::Array(v) | LuauTableEntry::Value(v) => collect_expr_identifiers(v, out),
                    LuauTableEntry::Record(_, v) => collect_expr_identifiers(v, out),
                    LuauTableEntry::General(k, v) => {
                        collect_expr_identifiers(k, out);
                        collect_expr_identifiers(v, out);
                    }
                }
            }
        }
        _ => {}
    }
}

fn count_expr_ident(expr: &LuauExpr, ident: &str) -> usize {
    match expr {
        LuauExpr::Identifier(id) if *id == ident => 1,
        LuauExpr::Unary { expr, .. } => count_expr_ident(expr, ident),
        LuauExpr::Binary { left, right, .. } => count_expr_ident(left, ident) + count_expr_ident(right, ident),
        LuauExpr::Index { table, key } => count_expr_ident(table, ident) + count_expr_ident(key, ident),
        LuauExpr::Call { func, args } => {
            count_expr_ident(func, ident) + args.iter().map(|a| count_expr_ident(a, ident)).sum::<usize>()
        }
        LuauExpr::MethodCall { target, args, .. } => {
            count_expr_ident(target, ident) + args.iter().map(|a| count_expr_ident(a, ident)).sum::<usize>()
        }
        LuauExpr::IfExp { cond, then_expr, else_expr } => {
            count_expr_ident(cond, ident) + count_expr_ident(then_expr, ident) + count_expr_ident(else_expr, ident)
        }
        LuauExpr::Table(entries) => entries
            .iter()
            .map(|e| match e {
                LuauTableEntry::Array(v) | LuauTableEntry::Value(v) => count_expr_ident(v, ident),
                LuauTableEntry::Record(_, v) => count_expr_ident(v, ident),
                LuauTableEntry::General(k, v) => count_expr_ident(k, ident) + count_expr_ident(v, ident),
            })
            .sum(),
        LuauExpr::Function { params, body, .. } => {
            if params.contains(&ident) {
                0
            } else {
                body.iter().map(|s| count_stat_ident(s, ident)).sum()
            }
        }
        _ => 0,
    }
}

fn count_stat_ident(stat: &LuauStat, ident: &str) -> usize {
    match stat {
        LuauStat::LocalAssign { values, .. } => values.iter().map(|v| count_expr_ident(v, ident)).sum(),
        LuauStat::Assign { targets, values } => {
            targets.iter().map(|t| count_expr_ident(t, ident)).sum::<usize>()
                + values.iter().map(|v| count_expr_ident(v, ident)).sum::<usize>()
        }
        LuauStat::CompoundAssign { target, value, .. } => {
            count_expr_ident(target, ident) + count_expr_ident(value, ident)
        }
        LuauStat::Call(expr) => count_expr_ident(expr, ident),
        LuauStat::Return(vals) => vals.iter().map(|v| count_expr_ident(v, ident)).sum(),
        LuauStat::If { cond, then_branch, elseif_branches, else_branch } => {
            count_expr_ident(cond, ident)
                + then_branch.iter().map(|s| count_stat_ident(s, ident)).sum::<usize>()
                + elseif_branches
                    .iter()
                    .map(|b| count_expr_ident(&b.cond, ident) + b.body.iter().map(|s| count_stat_ident(s, ident)).sum::<usize>())
                    .sum::<usize>()
                + else_branch.as_ref().map_or(0, |b| b.iter().map(|s| count_stat_ident(s, ident)).sum())
        }
        LuauStat::While { cond, body } | LuauStat::Repeat { cond, body } => {
            count_expr_ident(cond, ident) + body.iter().map(|s| count_stat_ident(s, ident)).sum::<usize>()
        }
        LuauStat::NumericFor { var, start, limit, step, body } => {
            count_expr_ident(start, ident)
                + count_expr_ident(limit, ident)
                + step.as_ref().map_or(0, |s| count_expr_ident(s, ident))
                + if *var == ident { 0 } else { body.iter().map(|s| count_stat_ident(s, ident)).sum::<usize>() }
        }
        LuauStat::GenericFor { vars, iterators, body } => {
            iterators.iter().map(|it| count_expr_ident(it, ident)).sum::<usize>()
                + if vars.contains(&ident) { 0 } else { body.iter().map(|s| count_stat_ident(s, ident)).sum::<usize>() }
        }
        LuauStat::LocalFunctionDef { name, params, body, .. } => {
            if *name == ident || params.contains(&ident) {
                0
            } else {
                body.iter().map(|s| count_stat_ident(s, ident)).sum()
            }
        }
        LuauStat::FunctionDef { name, params, body, .. } => {
            if *name == ident || params.contains(&ident) {
                0
            } else {
                body.iter().map(|s| count_stat_ident(s, ident)).sum()
            }
        }
        LuauStat::DoBlock(body) => body.iter().map(|s| count_stat_ident(s, ident)).sum(),
        _ => 0,
    }
}

fn substitute_expr_ident<'arena>(
    expr: &mut LuauExpr<'arena>,
    ident: &str,
    replacement: &LuauExpr<'arena>,
) -> usize {
    if let LuauExpr::Identifier(id) = expr {
        if *id == ident {
            *expr = replacement.clone();
            return 1;
        }
    }

    match expr {
        LuauExpr::Unary { expr: inner, .. } => substitute_expr_ident(inner, ident, replacement),
        LuauExpr::Binary { left, right, .. } => {
            substitute_expr_ident(left, ident, replacement) + substitute_expr_ident(right, ident, replacement)
        }
        LuauExpr::Index { table, key } => {
            substitute_expr_ident(table, ident, replacement) + substitute_expr_ident(key, ident, replacement)
        }
        LuauExpr::Call { func, args } => {
            let mut count = substitute_expr_ident(func, ident, replacement);
            for arg in args {
                count += substitute_expr_ident(arg, ident, replacement);
            }
            count
        }
        LuauExpr::MethodCall { target, args, .. } => {
            let mut count = substitute_expr_ident(target, ident, replacement);
            for arg in args {
                count += substitute_expr_ident(arg, ident, replacement);
            }
            count
        }
        LuauExpr::IfExp { cond, then_expr, else_expr } => {
            substitute_expr_ident(cond, ident, replacement)
                + substitute_expr_ident(then_expr, ident, replacement)
                + substitute_expr_ident(else_expr, ident, replacement)
        }
        LuauExpr::Table(entries) => {
            let mut count = 0;
            for entry in entries {
                match entry {
                    LuauTableEntry::Array(v) | LuauTableEntry::Value(v) => {
                        count += substitute_expr_ident(v, ident, replacement);
                    }
                    LuauTableEntry::Record(_, v) => {
                        count += substitute_expr_ident(v, ident, replacement);
                    }
                    LuauTableEntry::General(k, v) => {
                        count += substitute_expr_ident(k, ident, replacement);
                        count += substitute_expr_ident(v, ident, replacement);
                    }
                }
            }
            count
        }
        LuauExpr::Function { params, body, .. } => {
            if params.contains(&ident) {
                0
            } else {
                body.iter_mut().map(|s| substitute_stat_ident(s, ident, replacement)).sum()
            }
        }
        _ => 0,
    }
}

fn substitute_stat_ident<'arena>(
    stat: &mut LuauStat<'arena>,
    ident: &str,
    replacement: &LuauExpr<'arena>,
) -> usize {
    match stat {
        LuauStat::LocalAssign { values, .. } => {
            values.iter_mut().map(|v| substitute_expr_ident(v, ident, replacement)).sum()
        }
        LuauStat::Assign { targets, values } => {
            // In targets, only substitute within table index keys/tables, not bare identifiers!
            let target_count: usize = targets
                .iter_mut()
                .map(|t| {
                    if let LuauExpr::Index { table, key } = t {
                        substitute_expr_ident(table, ident, replacement)
                            + substitute_expr_ident(key, ident, replacement)
                    } else {
                        0
                    }
                })
                .sum();
            target_count + values.iter_mut().map(|v| substitute_expr_ident(v, ident, replacement)).sum::<usize>()
        }
        LuauStat::CompoundAssign { target, value, .. } => {
            let target_count = if let LuauExpr::Index { table, key } = target {
                substitute_expr_ident(table, ident, replacement)
                    + substitute_expr_ident(key, ident, replacement)
            } else {
                0
            };
            target_count + substitute_expr_ident(value, ident, replacement)
        }
        LuauStat::Call(expr) => substitute_expr_ident(expr, ident, replacement),
        LuauStat::Return(vals) => vals.iter_mut().map(|v| substitute_expr_ident(v, ident, replacement)).sum(),
        LuauStat::If { cond, then_branch, elseif_branches, else_branch } => {
            let mut count = substitute_expr_ident(cond, ident, replacement);
            for s in then_branch {
                count += substitute_stat_ident(s, ident, replacement);
            }
            for b in elseif_branches {
                count += substitute_expr_ident(&mut b.cond, ident, replacement);
                for s in &mut b.body {
                    count += substitute_stat_ident(s, ident, replacement);
                }
            }
            if let Some(els) = else_branch {
                for s in els {
                    count += substitute_stat_ident(s, ident, replacement);
                }
            }
            count
        }
        LuauStat::While { cond, body } | LuauStat::Repeat { cond, body } => {
            substitute_expr_ident(cond, ident, replacement)
                + body.iter_mut().map(|s| substitute_stat_ident(s, ident, replacement)).sum::<usize>()
        }
        LuauStat::NumericFor { var, start, limit, step, body } => {
            let mut count = substitute_expr_ident(start, ident, replacement)
                + substitute_expr_ident(limit, ident, replacement)
                + step.as_mut().map_or(0, |s| substitute_expr_ident(s, ident, replacement));
            if *var != ident {
                for s in body {
                    count += substitute_stat_ident(s, ident, replacement);
                }
            }
            count
        }
        LuauStat::GenericFor { vars, iterators, body } => {
            let mut count: usize = iterators.iter_mut().map(|it| substitute_expr_ident(it, ident, replacement)).sum();
            if !vars.contains(&ident) {
                for s in body {
                    count += substitute_stat_ident(s, ident, replacement);
                }
            }
            count
        }
        LuauStat::LocalFunctionDef { name, params, body, .. } => {
            if *name != ident && !params.contains(&ident) {
                body.iter_mut().map(|s| substitute_stat_ident(s, ident, replacement)).sum()
            } else {
                0
            }
        }
        LuauStat::FunctionDef { name, params, body, .. } => {
            if *name != ident && !params.contains(&ident) {
                body.iter_mut().map(|s| substitute_stat_ident(s, ident, replacement)).sum()
            } else {
                0
            }
        }
        LuauStat::DoBlock(body) => body.iter_mut().map(|s| substitute_stat_ident(s, ident, replacement)).sum(),
        _ => 0,
    }
}

fn is_tail_multret_position(stat: &LuauStat, target_name: &str) -> bool {
    match stat {
        LuauStat::Return(vals) => {
            if let Some(last) = vals.last() {
                expr_references_ident(last, target_name)
            } else {
                false
            }
        }
        LuauStat::Call(LuauExpr::Call { args, .. })
        | LuauStat::Call(LuauExpr::MethodCall { args, .. }) => {
            if let Some(last) = args.last() {
                expr_references_ident(last, target_name)
            } else {
                false
            }
        }
        _ => false,
    }
}

// ============================================================================
// Pass 7: Opaque Predicate & Branch Simplification
// ============================================================================

/// Inverts a comparison or logical condition expression.
pub fn negate_condition<'arena>(cond: LuauExpr<'arena>) -> LuauExpr<'arena> {
    match cond {
        LuauExpr::Binary { op, left, right } => {
            let inverted_op = match op {
                BinaryOp::Equal => Some(BinaryOp::NotEqual),
                BinaryOp::NotEqual => Some(BinaryOp::Equal),
                BinaryOp::Less => Some(BinaryOp::GreaterEq),
                BinaryOp::LessEq => Some(BinaryOp::Greater),
                BinaryOp::Greater => Some(BinaryOp::LessEq),
                BinaryOp::GreaterEq => Some(BinaryOp::Less),
                _ => None,
            };
            if let Some(inv) = inverted_op {
                LuauExpr::Binary {
                    op: inv,
                    left,
                    right,
                }
            } else {
                LuauExpr::Unary {
                    op: UnaryOp::Not,
                    expr: Box::new(LuauExpr::Binary { op, left, right }),
                }
            }
        }
        LuauExpr::Unary {
            op: UnaryOp::Not,
            expr,
        } => *expr,
        other => LuauExpr::Unary {
            op: UnaryOp::Not,
            expr: Box::new(other),
        },
    }
}

/// Checks whether an expression contains potential runtime side effects (function calls).
pub fn expr_has_side_effects<'arena>(expr: &LuauExpr<'arena>) -> bool {
    match expr {
        LuauExpr::Call { .. } | LuauExpr::MethodCall { .. } => true,
        LuauExpr::Unary { expr, .. } => expr_has_side_effects(expr),
        LuauExpr::Binary { left, right, .. } => {
            expr_has_side_effects(left) || expr_has_side_effects(right)
        }
        LuauExpr::Index { table, key } => {
            expr_has_side_effects(table) || expr_has_side_effects(key)
        }
        LuauExpr::IfExp {
            cond,
            then_expr,
            else_expr,
        } => {
            expr_has_side_effects(cond)
                || expr_has_side_effects(then_expr)
                || expr_has_side_effects(else_expr)
        }
        LuauExpr::Table(entries) => entries.iter().any(|e| match e {
            LuauTableEntry::Array(v) | LuauTableEntry::Value(v) => expr_has_side_effects(v),
            LuauTableEntry::Record(_, v) => expr_has_side_effects(v),
            LuauTableEntry::General(k, v) => {
                expr_has_side_effects(k) || expr_has_side_effects(v)
            }
        }),
        _ => false,
    }
}

/// Extracts all function call expressions from `expr` and appends them as statements.
pub fn extract_calls_from_expr<'arena>(expr: LuauExpr<'arena>, out: &mut Vec<LuauStat<'arena>>) {
    match expr {
        call @ LuauExpr::Call { .. } | call @ LuauExpr::MethodCall { .. } => {
            out.push(LuauStat::Call(call));
        }
        LuauExpr::Unary { expr, .. } => extract_calls_from_expr(*expr, out),
        LuauExpr::Binary { left, right, .. } => {
            extract_calls_from_expr(*left, out);
            extract_calls_from_expr(*right, out);
        }
        LuauExpr::Index { table, key } => {
            extract_calls_from_expr(*table, out);
            extract_calls_from_expr(*key, out);
        }
        LuauExpr::IfExp {
            cond,
            then_expr,
            else_expr,
        } => {
            extract_calls_from_expr(*cond, out);
            extract_calls_from_expr(*then_expr, out);
            extract_calls_from_expr(*else_expr, out);
        }
        LuauExpr::Table(entries) => {
            for e in entries {
                match e {
                    LuauTableEntry::Array(v) | LuauTableEntry::Value(v) => {
                        extract_calls_from_expr(v, out)
                    }
                    LuauTableEntry::Record(_, v) => extract_calls_from_expr(v, out),
                    LuauTableEntry::General(k, v) => {
                        extract_calls_from_expr(k, out);
                        extract_calls_from_expr(v, out);
                    }
                }
            }
        }
        _ => {}
    }
}

/// Pass A: Inverts `if cond then else ... end` when `then_branch` is empty.
pub fn simplify_empty_branches<'arena>(stats: Vec<LuauStat<'arena>>) -> Vec<LuauStat<'arena>> {
    stats.into_iter().map(simplify_stat_empty_branches).collect()
}

fn simplify_stat_empty_branches<'arena>(stat: LuauStat<'arena>) -> LuauStat<'arena> {
    match stat {
        LuauStat::If {
            cond,
            then_branch,
            elseif_branches,
            else_branch,
        } => {
            let then_branch = simplify_empty_branches(then_branch);
            let elseif_branches: Vec<ElseIfBranch<'arena>> = elseif_branches
                .into_iter()
                .map(|b| ElseIfBranch {
                    cond: b.cond,
                    body: simplify_empty_branches(b.body),
                })
                .collect();
            let mut else_branch = else_branch.map(simplify_empty_branches);

            // Invert condition if then_branch is empty and else_branch is present
            if then_branch.is_empty() && elseif_branches.is_empty() {
                if let Some(els) = else_branch.take() {
                    if !els.is_empty() {
                        return LuauStat::If {
                            cond: negate_condition(cond),
                            then_branch: els,
                            elseif_branches: Vec::new(),
                            else_branch: None,
                        };
                    }
                }
            }

            LuauStat::If {
                cond,
                then_branch,
                elseif_branches,
                else_branch,
            }
        }
        LuauStat::While { cond, body } => LuauStat::While {
            cond,
            body: simplify_empty_branches(body),
        },
        LuauStat::Repeat { cond, body } => LuauStat::Repeat {
            cond,
            body: simplify_empty_branches(body),
        },
        LuauStat::NumericFor {
            var,
            start,
            limit,
            step,
            body,
        } => LuauStat::NumericFor {
            var,
            start,
            limit,
            step,
            body: simplify_empty_branches(body),
        },
        LuauStat::GenericFor {
            vars,
            iterators,
            body,
        } => LuauStat::GenericFor {
            vars,
            iterators,
            body: simplify_empty_branches(body),
        },
        LuauStat::LocalFunctionDef {
            name,
            params,
            is_vararg,
            body,
        } => LuauStat::LocalFunctionDef {
            name,
            params,
            is_vararg,
            body: simplify_empty_branches(body),
        },
        LuauStat::FunctionDef {
            name,
            params,
            is_vararg,
            body,
        } => LuauStat::FunctionDef {
            name,
            params,
            is_vararg,
            body: simplify_empty_branches(body),
        },
        LuauStat::DoBlock(body) => LuauStat::DoBlock(simplify_empty_branches(body)),
        other => other,
    }
}

/// Pass B: Recursively prunes dead empty if statements and empty branch ladders.
pub fn prune_dead_branches<'arena>(mut stats: Vec<LuauStat<'arena>>) -> Vec<LuauStat<'arena>> {
    let mut prev_len = usize::MAX;
    let mut iterations = 0;
    while stats.len() != prev_len && iterations < 16 {
        prev_len = stats.len();
        iterations += 1;
        stats = prune_stats_pass(stats);
    }
    stats
}

fn prune_stats_pass<'arena>(stats: Vec<LuauStat<'arena>>) -> Vec<LuauStat<'arena>> {
    let mut out = Vec::with_capacity(stats.len());
    for stat in stats {
        match stat {
            LuauStat::If {
                cond,
                mut then_branch,
                mut elseif_branches,
                mut else_branch,
            } => {
                then_branch = prune_dead_branches(then_branch);
                for b in &mut elseif_branches {
                    b.body = prune_dead_branches(std::mem::take(&mut b.body));
                }
                elseif_branches.retain(|b| !b.body.is_empty());

                if let Some(els) = else_branch.take() {
                    let pruned_els = prune_dead_branches(els);
                    if !pruned_els.is_empty() {
                        else_branch = Some(pruned_els);
                    }
                }

                // If entire If statement has zero statements in all branches
                if then_branch.is_empty() && elseif_branches.is_empty() && else_branch.is_none() {
                    extract_calls_from_expr(cond, &mut out);
                    continue;
                }

                // If then_branch is empty but has non-empty elseif branches:
                // Promote the first elseif branch if cond has no side effects
                if then_branch.is_empty() && !elseif_branches.is_empty() {
                    if !expr_has_side_effects(&cond) {
                        let first = elseif_branches.remove(0);
                        out.push(LuauStat::If {
                            cond: first.cond,
                            then_branch: first.body,
                            elseif_branches,
                            else_branch,
                        });
                        continue;
                    }
                }

                out.push(LuauStat::If {
                    cond,
                    then_branch,
                    elseif_branches,
                    else_branch,
                });
            }
            LuauStat::While { cond, body } => {
                let pruned_body = prune_dead_branches(body);
                if pruned_body.is_empty() && !expr_has_side_effects(&cond) {
                    continue;
                }
                out.push(LuauStat::While {
                    cond,
                    body: pruned_body,
                });
            }
            LuauStat::Repeat { cond, body } => {
                out.push(LuauStat::Repeat {
                    cond,
                    body: prune_dead_branches(body),
                });
            }
            LuauStat::NumericFor {
                var,
                start,
                limit,
                step,
                body,
            } => {
                let pruned_body = prune_dead_branches(body);
                if pruned_body.is_empty()
                    && !expr_has_side_effects(&start)
                    && !expr_has_side_effects(&limit)
                    && step.as_ref().map_or(true, |s| !expr_has_side_effects(s))
                {
                    continue;
                }
                out.push(LuauStat::NumericFor {
                    var,
                    start,
                    limit,
                    step,
                    body: pruned_body,
                });
            }
            LuauStat::GenericFor {
                vars,
                iterators,
                body,
            } => {
                let pruned_body = prune_dead_branches(body);
                if pruned_body.is_empty()
                    && iterators.iter().all(|it| !expr_has_side_effects(it))
                {
                    continue;
                }
                out.push(LuauStat::GenericFor {
                    vars,
                    iterators,
                    body: pruned_body,
                });
            }
            LuauStat::DoBlock(body) => {
                let pruned_body = prune_dead_branches(body);
                if pruned_body.is_empty() {
                    continue;
                }
                out.push(LuauStat::DoBlock(pruned_body));
            }
            LuauStat::LocalFunctionDef {
                name,
                params,
                is_vararg,
                body,
            } => {
                out.push(LuauStat::LocalFunctionDef {
                    name,
                    params,
                    is_vararg,
                    body: prune_dead_branches(body),
                });
            }
            LuauStat::FunctionDef {
                name,
                params,
                is_vararg,
                body,
            } => {
                out.push(LuauStat::FunctionDef {
                    name,
                    params,
                    is_vararg,
                    body: prune_dead_branches(body),
                });
            }
            other => out.push(other),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use bumpalo::Bump;

    #[test]
    fn test_inline_temporaries_single_use_fold() {
        let bump = Bump::new();
        // local r0 = game
        // local r1 = r0.GetService
        // local r2 = r0
        // local r3 = "Players"
        // r1(r2, r3)
        let stats = vec![
            LuauStat::LocalAssign {
                names: vec!["r0"],
                values: vec![LuauExpr::Identifier("game")],
            },
            LuauStat::LocalAssign {
                names: vec!["r1"],
                values: vec![LuauExpr::Index {
                    table: Box::new(LuauExpr::Identifier("r0")),
                    key: Box::new(LuauExpr::String("GetService")),
                }],
            },
            LuauStat::LocalAssign {
                names: vec!["r2"],
                values: vec![LuauExpr::Identifier("r0")],
            },
            LuauStat::LocalAssign {
                names: vec!["r3"],
                values: vec![LuauExpr::String("Players")],
            },
            LuauStat::Call(LuauExpr::Call {
                func: Box::new(LuauExpr::Identifier("r1")),
                args: vec![LuauExpr::Identifier("r2"), LuauExpr::Identifier("r3")],
            }),
        ];

        let pipe = IdiomPipeline::default();
        let transformed = pipe.transform(stats, &bump);

        assert_eq!(transformed.len(), 1);
        match &transformed[0] {
            LuauStat::Call(LuauExpr::MethodCall { target, method, args }) => {
                assert_eq!(**target, LuauExpr::Identifier("game"));
                assert_eq!(*method, "GetService");
                assert_eq!(args.len(), 1);
                assert_eq!(args[0], LuauExpr::String("Players"));
            }
            other => panic!("Expected game:GetService(\"Players\"), got {other:?}"),
        }
    }

    #[test]
    fn test_inline_temporaries_multret_protection() {
        let bump = Bump::new();
        // local x = f()
        // return x
        // x in tail position of return should NOT inline f() directly without preserving truncation
        let stats = vec![
            LuauStat::LocalAssign {
                names: vec!["x"],
                values: vec![LuauExpr::Call {
                    func: Box::new(LuauExpr::Identifier("f")),
                    args: vec![],
                }],
            },
            LuauStat::Return(vec![LuauExpr::Identifier("x")]),
        ];

        let inlined = inline_temporaries(stats.clone(), &bump);
        assert_eq!(inlined.len(), 2, "Multret tail call should not be inlined into return");
    }

    #[test]
    fn test_prune_dead_locals_pure_vs_side_effects() {
        let stats = vec![
            LuauStat::LocalAssign {
                names: vec!["fn_dead"],
                values: vec![LuauExpr::Identifier("fn_40")],
            },
            LuauStat::LocalAssign {
                names: vec!["unused_call_ret"],
                values: vec![LuauExpr::Call {
                    func: Box::new(LuauExpr::Identifier("print")),
                    args: vec![LuauExpr::String("hello")],
                }],
            },
        ];

        let pruned = prune_dead_locals(stats);
        assert_eq!(pruned.len(), 1);
        match &pruned[0] {
            LuauStat::Call(LuauExpr::Call { func, args }) => {
                assert_eq!(**func, LuauExpr::Identifier("print"));
                assert_eq!(args.len(), 1);
            }
            other => panic!("Expected standalone print call, got {other:?}"),
        }
    }

    #[test]
    fn test_simplify_empty_branches_inversion() {
        // if a ~= b then else print("ok") end -> if a == b then print("ok") end
        let stat = LuauStat::If {
            cond: LuauExpr::Binary {
                op: BinaryOp::NotEqual,
                left: Box::new(LuauExpr::Identifier("a")),
                right: Box::new(LuauExpr::Identifier("b")),
            },
            then_branch: vec![],
            elseif_branches: vec![],
            else_branch: Some(vec![LuauStat::Call(LuauExpr::Call {
                func: Box::new(LuauExpr::Identifier("print")),
                args: vec![LuauExpr::String("ok")],
            })]),
        };

        let simplified = simplify_empty_branches(vec![stat]);
        assert_eq!(simplified.len(), 1);
        match &simplified[0] {
            LuauStat::If { cond, then_branch, else_branch, .. } => {
                assert!(else_branch.is_none());
                assert_eq!(then_branch.len(), 1);
                assert_eq!(
                    *cond,
                    LuauExpr::Binary {
                        op: BinaryOp::Equal,
                        left: Box::new(LuauExpr::Identifier("a")),
                        right: Box::new(LuauExpr::Identifier("b")),
                    }
                );
            }
            other => panic!("Expected inverted If, got {other:?}"),
        }
    }

    #[test]
    fn test_prune_dead_branches_empty_cascades() {
        // if cond then elseif cond2 then end
        let stat = LuauStat::If {
            cond: LuauExpr::Identifier("c1"),
            then_branch: vec![],
            elseif_branches: vec![
                ElseIfBranch {
                    cond: LuauExpr::Identifier("c2"),
                    body: vec![],
                },
            ],
            else_branch: None,
        };

        let pruned = prune_dead_branches(vec![stat]);
        assert_eq!(pruned.len(), 0, "All-empty if/elseif block should be completely pruned");
    }
}


