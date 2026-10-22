//! Luau 200-Local Budget Limiter & Scope Cut Splitting Engine.
//!
//! Enforces Luau VM local variable limit (<= 180 budget per stack frame)
//! using Difference-Array Prefix Sweep Optimization and DoBlock wrapping.

use std::collections::{HashMap, HashSet};
use crate::codegen::{LuauExpr, LuauStat};

/// Luau VM architectural limit.
pub const LUAU_MAX_LOCALS: usize = 200;
/// Safety threshold reserving 20 slots for temporary expressions.
pub const LUAU_LOCAL_BUDGET: usize = 180;

/// Returns all local variable identifiers declared by this statement.
pub fn declared_here<'a, 'arena>(stat: &'a LuauStat<'arena>) -> Vec<&'arena str> {
    match stat {
        LuauStat::LocalDecl(names) => names.clone(),
        LuauStat::LocalAssign { names, .. } => names.clone(),
        LuauStat::LocalFunctionDef { name, .. } => vec![*name],
        _ => Vec::new(),
    }
}

/// Returns declared identifiers that can be hoisted into preceding bare declarations.
pub fn hoistable_names<'a, 'arena>(stat: &'a LuauStat<'arena>) -> Vec<&'arena str> {
    match stat {
        LuauStat::LocalDecl(names) => names.clone(),
        LuauStat::LocalAssign { names, .. } => names.clone(),
        LuauStat::LocalFunctionDef { name, .. } => vec![*name],
        _ => Vec::new(),
    }
}

/// Recursively collects all variable identifiers referenced in an expression.
pub fn collect_expr_identifiers<'arena>(expr: &LuauExpr<'arena>, out: &mut HashSet<&'arena str>) {
    match expr {
        LuauExpr::Identifier(name) => {
            out.insert(*name);
        }
        LuauExpr::Unary { expr, .. } => {
            collect_expr_identifiers(expr, out);
        }
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
            for arg in args {
                collect_expr_identifiers(arg, out);
            }
        }
        LuauExpr::MethodCall { target, args, .. } => {
            collect_expr_identifiers(target, out);
            for arg in args {
                collect_expr_identifiers(arg, out);
            }
        }
        LuauExpr::IfExp {
            cond,
            then_expr,
            else_expr,
        } => {
            collect_expr_identifiers(cond, out);
            collect_expr_identifiers(then_expr, out);
            collect_expr_identifiers(else_expr, out);
        }
        LuauExpr::Function { params: _, is_vararg: _, body } => {
            for s in body {
                out.extend(names_anywhere(s));
            }
        }
        LuauExpr::Table(entries) => {
            for entry in entries {
                match entry {
                    crate::codegen::LuauTableEntry::Array(val)
                    | crate::codegen::LuauTableEntry::Value(val) => {
                        collect_expr_identifiers(val, out);
                    }
                    crate::codegen::LuauTableEntry::Record(_, val) => {
                        collect_expr_identifiers(val, out);
                    }
                    crate::codegen::LuauTableEntry::General(k, val) => {
                        collect_expr_identifiers(k, out);
                        collect_expr_identifiers(val, out);
                    }
                }
            }
        }
        _ => {}
    }
}

/// Recursively collects all variable identifiers referenced anywhere in a statement.
pub fn names_anywhere<'arena>(stat: &LuauStat<'arena>) -> HashSet<&'arena str> {
    let mut occ = HashSet::new();
    match stat {
        LuauStat::LocalDecl(names) => {
            for &n in names {
                occ.insert(n);
            }
        }
        LuauStat::LocalAssign { names, values } => {
            for &n in names {
                occ.insert(n);
            }
            for v in values {
                collect_expr_identifiers(v, &mut occ);
            }
        }
        LuauStat::LocalFunctionDef { name, params, body, .. } => {
            occ.insert(*name);
            for &p in params {
                occ.insert(p);
            }
            for s in body {
                occ.extend(names_anywhere(s));
            }
        }
        LuauStat::FunctionDef { name, params, body, .. } => {
            occ.insert(*name);
            for &p in params {
                occ.insert(p);
            }
            for s in body {
                occ.extend(names_anywhere(s));
            }
        }
        LuauStat::Assign { targets, values } => {
            for t in targets {
                collect_expr_identifiers(t, &mut occ);
            }
            for v in values {
                collect_expr_identifiers(v, &mut occ);
            }
        }
        LuauStat::CompoundAssign { target, value, .. } => {
            collect_expr_identifiers(target, &mut occ);
            collect_expr_identifiers(value, &mut occ);
        }
        LuauStat::Call(expr) => {
            collect_expr_identifiers(expr, &mut occ);
        }
        LuauStat::Return(vals) => {
            for v in vals {
                collect_expr_identifiers(v, &mut occ);
            }
        }
        LuauStat::If { cond, then_branch, elseif_branches, else_branch } => {
            collect_expr_identifiers(cond, &mut occ);
            for s in then_branch {
                occ.extend(names_anywhere(s));
            }
            for branch in elseif_branches {
                collect_expr_identifiers(&branch.cond, &mut occ);
                for s in &branch.body {
                    occ.extend(names_anywhere(s));
                }
            }
            if let Some(els) = else_branch {
                for s in els {
                    occ.extend(names_anywhere(s));
                }
            }
        }
        LuauStat::While { cond, body } | LuauStat::Repeat { cond, body } => {
            collect_expr_identifiers(cond, &mut occ);
            for s in body {
                occ.extend(names_anywhere(s));
            }
        }
        LuauStat::NumericFor { start, limit, step, body, .. } => {
            collect_expr_identifiers(start, &mut occ);
            collect_expr_identifiers(limit, &mut occ);
            if let Some(s) = step {
                collect_expr_identifiers(s, &mut occ);
            }
            for s in body {
                occ.extend(names_anywhere(s));
            }
        }
        LuauStat::GenericFor { iterators, body, .. } => {
            for it in iterators {
                collect_expr_identifiers(it, &mut occ);
            }
            for s in body {
                occ.extend(names_anywhere(s));
            }
        }
        LuauStat::DoBlock(body) => {
            for s in body {
                occ.extend(names_anywhere(s));
            }
        }
        _ => {}
    }
    occ
}

/// Rewrites local declarations for hoisted variables into plain assignments.
pub fn unlocalize<'arena>(
    stmts: Vec<LuauStat<'arena>>,
    hoisted: &HashSet<&str>,
) -> Vec<LuauStat<'arena>> {
    let mut out = Vec::with_capacity(stmts.len());
    for st in stmts {
        match st {
            LuauStat::LocalDecl(names) => {
                let keep: Vec<&'arena str> = names
                    .into_iter()
                    .filter(|n| !hoisted.contains(*n))
                    .collect();
                if !keep.is_empty() {
                    out.push(LuauStat::LocalDecl(keep));
                }
            }
            LuauStat::LocalAssign { names, values } => {
                let (hoisted_names, keep_names): (Vec<_>, Vec<_>) =
                    names.into_iter().partition(|n| hoisted.contains(*n));

                if keep_names.is_empty() {
                    let targets = hoisted_names
                        .into_iter()
                        .map(LuauExpr::Identifier)
                        .collect();
                    out.push(LuauStat::Assign { targets, values });
                } else if hoisted_names.is_empty() {
                    out.push(LuauStat::LocalAssign { names: keep_names, values });
                } else {
                    out.push(LuauStat::LocalDecl(keep_names.clone()));
                    let targets = keep_names
                        .into_iter()
                        .chain(hoisted_names.into_iter())
                        .map(LuauExpr::Identifier)
                        .collect();
                    out.push(LuauStat::Assign { targets, values });
                }
            }
            other => out.push(other),
        }
    }
    out
}

/// Identifies the optimal cut point via difference-array sweep and bisects the block.
pub fn split_block<'arena>(
    stmts: Vec<LuauStat<'arena>>,
    room: usize,
) -> Option<Vec<LuauStat<'arena>>> {
    let n = stmts.len();
    if n < 2 {
        return None;
    }

    let decl_at: Vec<Vec<&'arena str>> = stmts.iter().map(declared_here).collect();
    let total: usize = decl_at.iter().map(|d| d.len()).sum();
    if total <= room {
        return Some(stmts);
    }

    let mut first: HashMap<&'arena str, usize> = HashMap::new();
    let mut last: HashMap<&'arena str, usize> = HashMap::new();
    let mut hoistable: HashSet<&'arena str> = HashSet::new();

    for (i, st) in stmts.iter().enumerate() {
        for nm in names_anywhere(st) {
            last.insert(nm, i);
        }
        for &nm in &decl_at[i] {
            first.entry(nm).or_insert(i);
        }
        for nm in hoistable_names(st) {
            hoistable.insert(nm);
        }
    }

    // Difference arrays for range updates
    let mut cross_d = vec![0isize; n + 2];
    let mut bad_d = vec![0isize; n + 2];

    let mut sorted_first: Vec<(&'arena str, usize)> = first.iter().map(|(&nm, &f)| (nm, f)).collect();
    sorted_first.sort_by_key(|&(nm, f)| (f, nm));

    for (nm, f) in sorted_first {
        let lst = *last.get(nm).unwrap_or(&f);
        if lst > f {
            cross_d[f + 1] += 1;
            cross_d[lst + 1] -= 1;
            if !hoistable.contains(nm) {
                bad_d[f + 1] += 1;
                bad_d[lst + 1] -= 1;
            }
        }
    }

    let mut best: Option<((usize, usize), usize)> = None;
    let mut cross: isize = 0;
    let mut bad: isize = 0;
    let mut left: usize = 0;

    for k in 1..n {
        cross += cross_d[k];
        bad += bad_d[k];
        left += decl_at[k - 1].len();

        if bad > 0 || left <= cross as usize || (cross as usize) >= room {
            continue;
        }

        let cross_u = cross as usize;
        let peak = left.max(cross_u + total - left);
        let cost = (peak, cross_u);

        match &best {
            None => best = Some((cost, k)),
            Some((best_cost, _)) if cost < *best_cost => {
                best = Some((cost, k));
            }
            _ => {}
        }
    }

    let (best_cost, k) = best?;
    if best_cost.0 >= total {
        return None;
    }

    // Identify hoisted crossing variables: f < k <= lst
    let mut hoisted: Vec<&'arena str> = first
        .iter()
        .filter(|(&nm, &f)| {
            let lst = *last.get(nm).unwrap_or(&f);
            f < k && k <= lst
        })
        .map(|(&nm, _)| nm)
        .collect();

    // Deterministic sort: birth index first, then identifier name
    hoisted.sort_by_key(|&nm| (first[nm], nm));

    let hoisted_set: HashSet<&str> = hoisted.iter().copied().collect();
    let room2 = room.saturating_sub(hoisted.len());

    let mut out = Vec::new();

    // Emit hoisted declarations in chunks of 10
    for chunk in hoisted.chunks(10) {
        out.push(LuauStat::LocalDecl(chunk.to_vec()));
    }

    // Split statements into prefix and suffix
    let mut left_stmts = stmts;
    let right_stmts = left_stmts.split_off(k);

    // Unlocalize left prefix and recursively fit inside DoBlock
    let unlocalized_left = unlocalize(left_stmts, &hoisted_set);
    let fitted_left = fit_locals(unlocalized_left, room2);
    out.push(LuauStat::DoBlock(fitted_left));

    // Recursively fit right suffix
    let fitted_right = fit_locals(right_stmts, room2);
    out.extend(fitted_right);

    Some(out)
}

/// Represents a candidate maximal closed segment of statements.
#[derive(Clone, Debug)]
pub struct ClosedSegment {
    pub start: usize,
    pub end: usize,
    pub decl_count: usize,
}

/// Partitions a statement sequence into maximal closed segments.
pub fn find_closed_segments<'arena>(
    stmts: &[LuauStat<'arena>],
    decl_at: &[Vec<&'arena str>],
    last: &HashMap<&'arena str, usize>,
    hoisted: &HashSet<&'arena str>,
) -> Vec<ClosedSegment> {
    let mut segs = Vec::new();
    let mut start = 0;
    let mut cnt = 0;
    let mut reach: isize = -1;

    for i in 0..stmts.len() {
        for &nm in &decl_at[i] {
            if hoisted.contains(nm) {
                continue;
            }
            cnt += 1;
            let lst = *last.get(nm).unwrap_or(&i) as isize;
            reach = reach.max(lst);
        }

        if (i as isize) >= reach {
            segs.push(ClosedSegment {
                start,
                end: i + 1,
                decl_count: cnt,
            });
            start = i + 1;
            cnt = 0;
            reach = -1;
        }
    }

    if start < stmts.len() {
        segs.push(ClosedSegment {
            start,
            end: stmts.len(),
            decl_count: cnt,
        });
    }

    segs
}

/// Groups maximal closed segments into DoBlocks, hoisting long-lived variables.
pub fn wrap_segments<'arena>(
    stmts: Vec<LuauStat<'arena>>,
    room: usize,
    chunk: Option<usize>,
    dry: bool,
) -> (Vec<LuauStat<'arena>>, usize) {
    let n = stmts.len();
    if n == 0 {
        return (stmts, 0);
    }

    let mut last: HashMap<&'arena str, usize> = HashMap::new();
    let mut decl_at: Vec<Vec<&'arena str>> = Vec::with_capacity(n);

    for (i, st) in stmts.iter().enumerate() {
        for nm in names_anywhere(st) {
            last.insert(nm, i);
        }
        decl_at.push(declared_here(st));
    }

    let mut life: Vec<(usize, &'arena str)> = Vec::new();
    for (i, names) in decl_at.iter().enumerate() {
        for &nm in names {
            let lst = *last.get(nm).unwrap_or(&i);
            life.push((lst.saturating_sub(i), nm));
        }
    }
    life.sort_by_key(|&(span, nm)| (std::cmp::Reverse(span), nm));

    let chunk_size = chunk.unwrap_or_else(|| {
        let c = room / 3;
        c.clamp(20, 80)
    });

    let mut hoisted: HashSet<&'arena str> = HashSet::new();
    let mut k = 0;
    let mut segs;

    loop {
        segs = find_closed_segments(&stmts, &decl_at, &last, &hoisted);
        let worst = segs.iter().map(|s| s.decl_count).max().unwrap_or(0);

        if worst <= chunk_size || k >= life.len() || hoisted.len() + chunk_size >= room {
            break;
        }

        let step = (life.len() / 50).max(1);
        for _ in 0..step {
            if k < life.len() {
                hoisted.insert(life[k].1);
                k += 1;
            }
        }
    }

    let wraps = |s: &ClosedSegment| -> bool {
        s.decl_count >= 1 && (s.end - s.start) > 1
    };

    if dry {
        let non_wrapped_sum: usize = segs.iter().filter(|s| !wraps(s)).map(|s| s.decl_count).sum();
        let max_inner: usize = segs.iter().filter(|s| wraps(s)).map(|s| s.decl_count).max().unwrap_or(0);
        let peak = hoisted.len() + non_wrapped_sum + max_inner;
        return (stmts, peak);
    }

    let mut out = Vec::new();
    let hoisted_set: HashSet<&str> = hoisted.iter().copied().collect();

    for seg in segs {
        let seg_slice = stmts[seg.start..seg.end].to_vec();
        let unlocalized = unlocalize(seg_slice, &hoisted_set);

        let pre_names: Vec<&'arena str> = (seg.start..seg.end)
            .flat_map(|i| &decl_at[i])
            .copied()
            .filter(|nm| hoisted.contains(nm))
            .collect();

        for chunk in pre_names.chunks(10) {
            out.push(LuauStat::LocalDecl(chunk.to_vec()));
        }

        if wraps(&seg) {
            out.push(LuauStat::DoBlock(unlocalized));
        } else {
            out.extend(unlocalized);
        }
    }

    let peak = out.iter().map(|s| declared_here(s).len()).sum();
    (out, peak)
}

/// Fits a block of statements into `room` active locals via wrap_segments or split_block.
pub fn fit_locals<'arena>(
    stmts: Vec<LuauStat<'arena>>,
    room: usize,
) -> Vec<LuauStat<'arena>> {
    let total_decls: usize = stmts.iter().map(|s| declared_here(s).len()).sum();
    if total_decls <= room {
        return stmts;
    }

    // Pass 1: Try wrap_segments with difference-array hoisting
    let (_, dry_peak) = wrap_segments(stmts.clone(), room, None, true);
    if dry_peak <= room {
        let (wrapped, _) = wrap_segments(stmts, room, None, false);
        return wrapped;
    }

    // Pass 2: Try optimal difference-array sweep cut
    if let Some(split) = split_block(stmts.clone(), room) {
        return split;
    }

    // Pass 3: Forced bisection at midpoint n / 2 with crossing variable hoisting
    if stmts.len() >= 2 {
        let k = stmts.len() / 2;
        let mut first: HashMap<&'arena str, usize> = HashMap::new();
        let mut last: HashMap<&'arena str, usize> = HashMap::new();
        for (i, st) in stmts.iter().enumerate() {
            for nm in names_anywhere(st) {
                last.insert(nm, i);
            }
            for nm in declared_here(st) {
                first.entry(nm).or_insert(i);
            }
        }
        let mut hoisted: Vec<&'arena str> = first
            .iter()
            .filter(|(&nm, &f)| {
                let lst = *last.get(nm).unwrap_or(&f);
                f < k && k <= lst
            })
            .map(|(&nm, _)| nm)
            .collect();
        hoisted.sort_by_key(|&nm| (first[nm], nm));
        let hoisted_set: HashSet<&str> = hoisted.iter().copied().collect();
        let room2 = room.saturating_sub(hoisted.len()).max(1);

        let mut left_stmts = stmts;
        let right_stmts = left_stmts.split_off(k);

        let unlocalized_left = unlocalize(left_stmts, &hoisted_set);
        let fitted_left = fit_locals(unlocalized_left, room2);
        let fitted_right = fit_locals(right_stmts, room2);

        let mut out = Vec::new();
        for chunk in hoisted.chunks(10) {
            out.push(LuauStat::LocalDecl(chunk.to_vec()));
        }
        out.push(LuauStat::DoBlock(fitted_left));
        out.push(LuauStat::DoBlock(fitted_right));
        return out;
    }

    // Pass 4: Tight chunk wrap fallback
    let (wrapped, _) = wrap_segments(stmts, room, Some(room / 4), false);
    wrapped
}

/// Recursively limits active local variables across all nested scopes to <= budget.
pub fn limit_locals<'arena>(
    mut stmts: Vec<LuauStat<'arena>>,
    outer: usize,
    budget: usize,
) -> Vec<LuauStat<'arena>> {
    let total: usize = stmts.iter().map(|s| declared_here(s).len()).sum();
    if outer + total > budget {
        let room = budget.saturating_sub(outer);
        stmts = fit_locals(stmts, room);
    }

    let mut active = outer;
    for st in &mut stmts {
        match st {
            LuauStat::If {
                then_branch,
                elseif_branches,
                else_branch,
                ..
            } => {
                let fitted_then = limit_locals(std::mem::take(then_branch), active, budget);
                *then_branch = fitted_then;
                for branch in elseif_branches {
                    let fitted_branch =
                        limit_locals(std::mem::take(&mut branch.body), active, budget);
                    branch.body = fitted_branch;
                }
                if let Some(els) = else_branch {
                    let fitted_els = limit_locals(std::mem::take(els), active, budget);
                    *els = fitted_els;
                }
            }
            LuauStat::LocalFunctionDef { body, params, .. } => {
                let fitted_body = limit_locals(std::mem::take(body), params.len(), budget);
                *body = fitted_body;
            }
            LuauStat::FunctionDef { body, params, .. } => {
                let fitted_body = limit_locals(std::mem::take(body), params.len(), budget);
                *body = fitted_body;
            }
            LuauStat::While { body, .. } | LuauStat::Repeat { body, .. } => {
                let fitted_body = limit_locals(std::mem::take(body), active, budget);
                *body = fitted_body;
            }
            LuauStat::NumericFor { body, .. } => {
                let fitted_body = limit_locals(std::mem::take(body), active + 1, budget);
                *body = fitted_body;
            }
            LuauStat::GenericFor { vars, body, .. } => {
                let fitted_body = limit_locals(std::mem::take(body), active + vars.len(), budget);
                *body = fitted_body;
            }
            LuauStat::DoBlock(body) => {
                let fitted_body = limit_locals(std::mem::take(body), active, budget);
                *body = fitted_body;
            }
            _ => {}
        }

        active += declared_here(st).len();
    }

    stmts
}
