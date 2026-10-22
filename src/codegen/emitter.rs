//! Luau Pretty Printer & Source Code Emitter.
//!
//! Emits formatted, syntactically compliant Luau source code with minimal
//! redundant parentheses by strictly adhering to a 10-level operator precedence table.

use crate::codegen::{LuauExpr, LuauStat, LuauTableEntry, UnaryOp};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IndentStyle {
    Spaces(usize),
    Tabs,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmitterConfig {
    pub indent_style: IndentStyle,
    pub max_table_inline_width: usize,
    pub emit_comments: bool,
    pub newline: &'static str,
}

impl Default for EmitterConfig {
    fn default() -> Self {
        Self {
            indent_style: IndentStyle::Spaces(4),
            max_table_inline_width: 80,
            emit_comments: true,
            newline: "\n",
        }
    }
}

pub struct LuauEmitter {
    pub config: EmitterConfig,
    pub buffer: String,
    pub current_indent: usize,
}

impl LuauEmitter {
    pub fn new(config: EmitterConfig) -> Self {
        Self {
            config,
            buffer: String::with_capacity(4096),
            current_indent: 0,
        }
    }

    /// Emits indentation pad for the current depth.
    fn emit_indent(&mut self) {
        match self.config.indent_style {
            IndentStyle::Spaces(n) => {
                let count = self.current_indent * n;
                for _ in 0..count {
                    self.buffer.push(' ');
                }
            }
            IndentStyle::Tabs => {
                for _ in 0..self.current_indent {
                    self.buffer.push('\t');
                }
            }
        }
    }

    /// Emits a statement with proper indentation and newline.
    pub fn emit_stat(&mut self, stat: &LuauStat) {
        match stat {
            LuauStat::LocalDecl(names) => {
                if !names.is_empty() {
                    self.emit_indent();
                    self.buffer.push_str("local ");
                    self.buffer.push_str(&names.join(", "));
                    self.buffer.push_str(self.config.newline);
                }
            }
            LuauStat::LocalAssign { names, values } => {
                if !names.is_empty() {
                    self.emit_indent();
                    self.buffer.push_str("local ");
                    self.buffer.push_str(&names.join(", "));
                    if !values.is_empty() {
                        self.buffer.push_str(" = ");
                        let val_strs: Vec<String> = values
                            .iter()
                            .map(|v| self.format_expr(v, 0))
                            .collect();
                        self.buffer.push_str(&val_strs.join(", "));
                    }
                    self.buffer.push_str(self.config.newline);
                }
            }
            LuauStat::LocalFunctionDef {
                name,
                params,
                is_vararg,
                body,
            } => {
                self.emit_indent();
                self.buffer.push_str(&format!("local function {name}("));
                self.emit_params(params, *is_vararg);
                self.buffer.push(')');
                self.buffer.push_str(self.config.newline);

                self.current_indent += 1;
                for st in body {
                    self.emit_stat(st);
                }
                self.current_indent -= 1;

                self.emit_indent();
                self.buffer.push_str("end");
                self.buffer.push_str(self.config.newline);
            }
            LuauStat::FunctionDef {
                name,
                params,
                is_vararg,
                body,
            } => {
                self.emit_indent();
                self.buffer.push_str(&format!("function {name}("));
                self.emit_params(params, *is_vararg);
                self.buffer.push(')');
                self.buffer.push_str(self.config.newline);

                self.current_indent += 1;
                for st in body {
                    self.emit_stat(st);
                }
                self.current_indent -= 1;

                self.emit_indent();
                self.buffer.push_str("end");
                self.buffer.push_str(self.config.newline);
            }
            LuauStat::CompoundAssign { target, op, value } => {
                self.emit_indent();
                let tgt_str = self.format_expr(target, 0);
                if tgt_str.starts_with('(') {
                    self.buffer.push(';');
                }
                self.buffer.push_str(&tgt_str);
                self.buffer.push(' ');
                self.buffer.push_str(op.as_str());
                self.buffer.push(' ');
                self.buffer.push_str(&self.format_expr(value, 0));
                self.buffer.push_str(self.config.newline);
            }
            LuauStat::Assign { targets, values } => {
                if !targets.is_empty() {
                    self.emit_indent();
                    let tgt_strs: Vec<String> = targets
                        .iter()
                        .map(|t| self.format_expr(t, 0))
                        .collect();
                    let val_strs: Vec<String> = values
                        .iter()
                        .map(|v| self.format_expr(v, 0))
                        .collect();
                    let joined_targets = tgt_strs.join(", ");
                    if joined_targets.starts_with('(') {
                        self.buffer.push(';');
                    }
                    self.buffer.push_str(&joined_targets);
                    self.buffer.push_str(" = ");
                    self.buffer.push_str(&val_strs.join(", "));
                    self.buffer.push_str(self.config.newline);
                }
            }
            LuauStat::Call(expr) => {
                self.emit_indent();
                let call_str = self.format_expr(expr, 0);
                if call_str.starts_with('(') {
                    self.buffer.push(';');
                }
                self.buffer.push_str(&call_str);
                self.buffer.push_str(self.config.newline);
            }

            LuauStat::Return(vals) => {
                self.emit_indent();
                if vals.is_empty() {
                    self.buffer.push_str("return");
                } else {
                    let val_strs: Vec<String> = vals
                        .iter()
                        .map(|v| self.format_expr(v, 0))
                        .collect();
                    self.buffer.push_str(&format!("return {}", val_strs.join(", ")));
                }
                self.buffer.push_str(self.config.newline);
            }
            LuauStat::If {
                cond,
                then_branch,
                elseif_branches,
                else_branch,
            } => {
                self.emit_indent();
                self.buffer
                    .push_str(&format!("if {} then", self.format_expr(cond, 0)));
                self.buffer.push_str(self.config.newline);

                self.current_indent += 1;
                for st in then_branch {
                    self.emit_stat(st);
                }
                self.current_indent -= 1;

                for branch in elseif_branches {
                    self.emit_indent();
                    self.buffer.push_str(&format!(
                        "elseif {} then",
                        self.format_expr(&branch.cond, 0)
                    ));
                    self.buffer.push_str(self.config.newline);
                    self.current_indent += 1;
                    for st in &branch.body {
                        self.emit_stat(st);
                    }
                    self.current_indent -= 1;
                }

                if let Some(els) = else_branch {
                    self.emit_indent();
                    self.buffer.push_str("else");
                    self.buffer.push_str(self.config.newline);
                    self.current_indent += 1;
                    for st in els {
                        self.emit_stat(st);
                    }
                    self.current_indent -= 1;
                }

                self.emit_indent();
                self.buffer.push_str("end");
                self.buffer.push_str(self.config.newline);
            }
            LuauStat::While { cond, body } => {
                self.emit_indent();
                self.buffer
                    .push_str(&format!("while {} do", self.format_expr(cond, 0)));
                self.buffer.push_str(self.config.newline);
                self.current_indent += 1;
                for st in body {
                    self.emit_stat(st);
                }
                self.current_indent -= 1;
                self.emit_indent();
                self.buffer.push_str("end");
                self.buffer.push_str(self.config.newline);
            }
            LuauStat::Repeat { cond, body } => {
                self.emit_indent();
                self.buffer.push_str("repeat");
                self.buffer.push_str(self.config.newline);
                self.current_indent += 1;
                for st in body {
                    self.emit_stat(st);
                }
                self.current_indent -= 1;
                self.emit_indent();
                self.buffer
                    .push_str(&format!("until {}", self.format_expr(cond, 0)));
                self.buffer.push_str(self.config.newline);
            }
            LuauStat::NumericFor {
                var,
                start,
                limit,
                step,
                body,
            } => {
                self.emit_indent();
                if let Some(s) = step {
                    self.buffer.push_str(&format!(
                        "for {var} = {}, {}, {} do",
                        self.format_expr(start, 0),
                        self.format_expr(limit, 0),
                        self.format_expr(s, 0)
                    ));
                } else {
                    self.buffer.push_str(&format!(
                        "for {var} = {}, {} do",
                        self.format_expr(start, 0),
                        self.format_expr(limit, 0)
                    ));
                }
                self.buffer.push_str(self.config.newline);
                self.current_indent += 1;
                for st in body {
                    self.emit_stat(st);
                }
                self.current_indent -= 1;
                self.emit_indent();
                self.buffer.push_str("end");
                self.buffer.push_str(self.config.newline);
            }
            LuauStat::GenericFor {
                vars,
                iterators,
                body,
            } => {
                self.emit_indent();
                let it_strs: Vec<String> = iterators
                    .iter()
                    .map(|it| self.format_expr(it, 0))
                    .collect();
                self.buffer.push_str(&format!(
                    "for {} in {} do",
                    vars.join(", "),
                    it_strs.join(", ")
                ));
                self.buffer.push_str(self.config.newline);
                self.current_indent += 1;
                for st in body {
                    self.emit_stat(st);
                }
                self.current_indent -= 1;
                self.emit_indent();
                self.buffer.push_str("end");
                self.buffer.push_str(self.config.newline);
            }
            LuauStat::DoBlock(body) => {
                self.emit_indent();
                self.buffer.push_str("do");
                self.buffer.push_str(self.config.newline);
                self.current_indent += 1;
                for st in body {
                    self.emit_stat(st);
                }
                self.current_indent -= 1;
                self.emit_indent();
                self.buffer.push_str("end");
                self.buffer.push_str(self.config.newline);
            }
            LuauStat::Break => {
                self.emit_indent();
                self.buffer.push_str("break");
                self.buffer.push_str(self.config.newline);
            }
            LuauStat::Continue => {
                self.emit_indent();
                self.buffer.push_str("continue");
                self.buffer.push_str(self.config.newline);
            }
            LuauStat::Comment(c) => {
                if self.config.emit_comments {
                    self.emit_indent();
                    self.buffer.push_str(&format!("-- {c}"));
                    self.buffer.push_str(self.config.newline);
                }
            }
            LuauStat::Raw(r) => {
                self.emit_indent();
                self.buffer.push_str(r);
                self.buffer.push_str(self.config.newline);
            }
        }
    }

    /// Formats an expression taking into account precedence context to minimize parens.
    pub fn format_expr(&self, expr: &LuauExpr, prec_ctx: u8) -> String {
        match expr {
            LuauExpr::Nil => "nil".to_string(),
            LuauExpr::Boolean(b) => b.to_string(),
            LuauExpr::Number(n) => {
                if n.is_nan() {
                    "(0/0)".to_string()
                } else if *n == f64::INFINITY {
                    "math.huge".to_string()
                } else if *n == f64::NEG_INFINITY {
                    "-math.huge".to_string()
                } else if n.fract() == 0.0 && n.is_finite() && n.abs() < 9e18 {
                    format!("{:.0}", n)
                } else {
                    format!("{n}")
                }
            }
            LuauExpr::String(s) => format!("\"{}\"", s.escape_default()),
            LuauExpr::Identifier(name) => {
                if is_valid_ident(name) {
                    name.to_string()
                } else {
                    format!("_G[\"{}\"]", name.escape_default())
                }
            }
            LuauExpr::Vararg => "...".to_string(),
            LuauExpr::Raw(r) => r.to_string(),

            LuauExpr::Unary { op, expr } => {
                let my_prec = op.precedence(); // Level 7
                let inner = self.format_expr(expr, my_prec);
                // Safe formatting: if op is Minus and inner starts with `-`, use parentheses to prevent `--` comment
                let s = if *op == UnaryOp::Negate && inner.starts_with('-') {
                    format!("-({inner})")
                } else {
                    format!("{}{inner}", op.as_str())
                };
                if my_prec < prec_ctx {
                    format!("({s})")
                } else {
                    s
                }
            }

            LuauExpr::Binary { op, left, right } => {
                let my_prec = op.precedence();
                let (left_ctx, right_ctx) = if op.is_right_associative() {
                    (my_prec + 1, my_prec)
                } else {
                    (my_prec, my_prec + 1)
                };

                let l_str = self.format_expr(left, left_ctx);
                let r_str = self.format_expr(right, right_ctx);
                let s = format!("{l_str} {} {r_str}", op.as_str());

                if my_prec < prec_ctx {
                    format!("({s})")
                } else {
                    s
                }
            }

            LuauExpr::Index { table, key } => {
                let tbl_str = self.format_prefix_expr(table);
                match key.as_ref() {
                    LuauExpr::String(prop) if is_valid_ident(prop) => {
                        format!("{tbl_str}.{prop}")
                    }
                    _ => {
                        let k_str = self.format_expr(key, 0);
                        format!("{tbl_str}[{k_str}]")
                    }
                }
            }

            LuauExpr::Call { func, args } => {
                let f_str = self.format_prefix_expr(func);
                let arg_strs: Vec<String> = args
                    .iter()
                    .map(|a| self.format_expr(a, 0))
                    .collect();
                format!("{f_str}({})", arg_strs.join(", "))
            }

            LuauExpr::MethodCall {
                target,
                method,
                args,
            } => {
                let tgt_str = self.format_prefix_expr(target);
                let arg_strs: Vec<String> = args
                    .iter()
                    .map(|a| self.format_expr(a, 0))
                    .collect();
                format!("{tgt_str}:{method}({})", arg_strs.join(", "))
            }

            LuauExpr::IfExp {
                cond,
                then_expr,
                else_expr,
            } => {
                let c = self.format_expr(cond, 0);
                let t = self.format_expr(then_expr, 0);
                let e = self.format_expr(else_expr, 0);
                let s = format!("if {c} then {t} else {e}");
                if prec_ctx > 0 {
                    format!("({s})")
                } else {
                    s
                }
            }

            LuauExpr::Table(entries) => self.format_table(entries),

            LuauExpr::Function {
                params,
                is_vararg,
                body,
            } => {
                let mut out = String::new();
                out.push_str("function(");
                let mut param_list: Vec<String> = params.iter().map(|p| p.to_string()).collect();
                if *is_vararg {
                    param_list.push("...".to_string());
                }
                out.push_str(&param_list.join(", "));
                out.push(')');
                out.push_str(self.config.newline);

                let mut sub_emitter = LuauEmitter {
                    config: self.config.clone(),
                    buffer: String::new(),
                    current_indent: self.current_indent + 1,
                };
                for st in body {
                    sub_emitter.emit_stat(st);
                }
                out.push_str(&sub_emitter.buffer);

                let pad = match self.config.indent_style {
                    IndentStyle::Spaces(n) => " ".repeat(self.current_indent * n),
                    IndentStyle::Tabs => "\t".repeat(self.current_indent),
                };
                out.push_str(&pad);
                out.push_str("end");
                out
            }
        }
    }

    /// Formats a prefix expression (primary expression before index, call, or method call).
    fn format_prefix_expr(&self, expr: &LuauExpr) -> String {
        match expr {
            LuauExpr::Identifier(_)
            | LuauExpr::Call { .. }
            | LuauExpr::MethodCall { .. }
            | LuauExpr::Index { .. } => self.format_expr(expr, 9),
            _ => format!("({})", self.format_expr(expr, 0)),
        }
    }

    /// Formats a table constructor, selecting single-line vs multiline.
    fn format_table(&self, entries: &[LuauTableEntry]) -> String {
        if entries.is_empty() {
            return "{}".to_string();
        }

        let parts: Vec<String> = entries
            .iter()
            .map(|entry| match entry {
                LuauTableEntry::Array(val) | LuauTableEntry::Value(val) => {
                    self.format_expr(val, 0)
                }
                LuauTableEntry::Record(key, val) => {
                    if is_valid_ident(key) {
                        format!("{key} = {}", self.format_expr(val, 0))
                    } else {
                        format!("[\"{}\"] = {}", key.escape_default(), self.format_expr(val, 0))
                    }
                }
                LuauTableEntry::General(k, v) => {
                    format!("[{}] = {}", self.format_expr(k, 0), self.format_expr(v, 0))
                }
            })
            .collect();

        let single_line = format!("{{ {} }}", parts.join(", "));
        if !single_line.contains('\n') && single_line.len() <= self.config.max_table_inline_width {
            return single_line;
        }

        // Multiline table formatting
        let pad_entry = match self.config.indent_style {
            IndentStyle::Spaces(n) => " ".repeat((self.current_indent + 1) * n),
            IndentStyle::Tabs => "\t".repeat(self.current_indent + 1),
        };
        let pad_close = match self.config.indent_style {
            IndentStyle::Spaces(n) => " ".repeat(self.current_indent * n),
            IndentStyle::Tabs => "\t".repeat(self.current_indent),
        };

        let mut out = String::from("{\n");
        for p in parts {
            out.push_str(&format!("{pad_entry}{p},\n"));
        }
        out.push_str(&format!("{pad_close}}}"));
        out
    }

    fn emit_params(&mut self, params: &[&str], is_vararg: bool) {
        let mut list: Vec<String> = params.iter().map(|p| p.to_string()).collect();
        if is_vararg {
            list.push("...".to_string());
        }
        self.buffer.push_str(&list.join(", "));
    }
}

/// Validates whether an identifier string is a valid Luau identifier name and not a reserved keyword.
pub fn is_valid_ident(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    const LUA_KEYWORDS: &[&str] = &[
        "and", "break", "do", "else", "elseif", "end", "false", "for", "function", "if", "in",
        "local", "nil", "not", "or", "repeat", "return", "then", "true", "until", "while",
        "continue",
    ];
    if LUA_KEYWORDS.contains(&s) {
        return false;
    }
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}
