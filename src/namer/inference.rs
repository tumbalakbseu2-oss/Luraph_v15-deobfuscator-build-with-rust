//! Roblox context semantic identifier inference engine.
//!
//! Analyzes typed IR expressions and instructions to infer idiomatic names
//! for datatypes, service singletons, instance queries, signal connections,
//! and global utilities using a formal Semantic Priority Lattice.

use crate::cfg::ir::{BinOp, IrExpr, UnOp};

/// Priority of inferred semantic naming hint.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum HintPriority {
    /// Default fallback (e.g. "v", "arg", "i").
    Default = 0,
    /// Weak type hint (e.g. "str", "n", "tbl", "flag", "value").
    Weak = 1,
    /// Contextual label from concatenation or table key (e.g. "health", "position").
    Label = 2,
    /// Strong semantic hint from Roblox API, service, or datatype (e.g. "players", "vector").
    Strong = 3,
}

/// A semantic naming recommendation with confidence and origin.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticHint {
    pub name: String,
    pub priority: HintPriority,
    pub origin_pc: usize,
}

impl SemanticHint {
    #[inline]
    pub fn new(name: impl Into<String>, priority: HintPriority, origin_pc: usize) -> Self {
        Self {
            name: name.into(),
            priority,
            origin_pc,
        }
    }

    /// Merges two hints, preserving the higher priority.
    /// If priorities are equal, the earlier definition/origin wins.
    pub fn merge(self, other: Self) -> Self {
        if self.priority > other.priority {
            self
        } else if other.priority > self.priority {
            other
        } else if self.origin_pc <= other.origin_pc {
            self
        } else {
            other
        }
    }
}

/// Luau reserved keywords and language identifiers.
pub const LUAU_KEYWORDS: &[&str] = &[
    "and", "break", "do", "else", "elseif", "end", "false", "for", "function", "if",
    "in", "local", "nil", "not", "or", "repeat", "return", "then", "true", "until",
    "while", "continue", "type", "export", "self",
];

/// Looks up standard Roblox datatype constructors.
pub fn lookup_datatype(name: &str) -> Option<&'static str> {
    match name {
        "Vector3" => Some("vector"),
        "Vector2" => Some("vector2"),
        "UDim2" => Some("udim2"),
        "UDim" => Some("udim"),
        "CFrame" => Some("cframe"),
        "Color3" => Some("color"),
        "TweenInfo" => Some("tweenInfo"),
        "ColorSequence" => Some("colorSequence"),
        "NumberSequence" => Some("numberSequence"),
        "NumberRange" => Some("numberRange"),
        "Rect" => Some("rect"),
        "RaycastParams" => Some("raycastParams"),
        "OverlapParams" => Some("overlapParams"),
        "Ray" => Some("ray"),
        "BrickColor" => Some("brickColor"),
        "Font" => Some("font"),
        "PhysicalProperties" => Some("physicalProperties"),
        "DateTime" => Some("dateTime"),
        "Random" => Some("random"),
        "PathWaypoint" => Some("pathWaypoint"),
        _ => None,
    }
}

/// Looks up standard Roblox instance and string methods.
pub fn lookup_method(method: &str) -> Option<&'static str> {
    match method {
        "GetChildren" => Some("children"),
        "GetDescendants" => Some("descendants"),
        "GetPlayers" => Some("players"),
        "Clone" => Some("clone"),
        "Connect" | "Once" | "ConnectParallel" => Some("connection"),
        "Create" => Some("tween"),
        "GetPropertyChangedSignal" => Some("signal"),
        "GetMouse" => Some("mouse"),
        "Raycast" => Some("hit"),
        "InvokeServer" | "InvokeClient" | "HttpGet" | "GetAsync" => Some("response"),
        "FireServer" | "FireClient" | "FireAllClients" => Some("response"),
        "WaitForChild" | "FindFirstChild" | "FindFirstChildOfClass" | "FindFirstChildWhichIsA" => Some("child"),
        "JSONDecode" => Some("data"),
        "JSONEncode" => Some("json"),
        "match" => Some("match"),
        "gsub" | "sub" | "lower" | "upper" | "format" | "rep" => Some("str"),
        "find" => Some("pos"),
        "split" => Some("parts"),
        "GetAttribute" => Some("attribute"),
        "IsA" => Some("isA"),
        "Wait" => Some("result"),
        _ => None,
    }
}

/// Looks up standard Luau global function return semantics.
pub fn lookup_global(name: &str) -> Option<&'static str> {
    match name {
        "tick" | "time" | "os.clock" | "os.time" => Some("now"),
        "type" | "typeof" => Some("kind"),
        "tostring" => Some("str"),
        "tonumber" => Some("num"),
        "setmetatable" => Some("obj"),
        "loadstring" => Some("chunk"),
        "getgenv" => Some("genv"),
        "getfenv" => Some("env"),
        "gethui" => Some("hui"),
        "string.format" | "string.rep" | "string.sub" | "table.concat" => Some("str"),
        "math.floor" | "math.clamp" | "math.min" | "math.max" | "math.abs" | "math.random" | "math.sqrt" => Some("n"),
        "select" | "unpack" | "table.unpack" | "rawget" => Some("val"),
        "next" => Some("key"),
        "coroutine.create" | "task.spawn" | "task.delay" | "task.defer" => Some("thread"),
        "Instance.new" => Some("instance"),
        "newproxy" => Some("proxy"),
        "Callback" | "callback" => Some("callback"),
        _ => None,
    }
}

/// Looks up standard RBXScriptSignal callback parameter names.
pub fn lookup_signal_param(signal: &str, param_idx: usize) -> Option<&'static str> {
    match signal {
        "PlayerAdded" | "PlayerRemoving" => (param_idx == 0).then_some("player"),
        "CharacterAdded" | "CharacterRemoving" | "CharacterAppearanceLoaded" => (param_idx == 0).then_some("character"),
        "Touched" | "TouchEnded" => (param_idx == 0).then_some("hit"),
        "InputBegan" | "InputEnded" | "InputChanged" => match param_idx {
            0 => Some("input"),
            1 => Some("gameProcessed"),
            _ => None,
        },
        "Heartbeat" | "RenderStepped" | "PreSimulation" | "PostSimulation" | "PreRender" => (param_idx == 0).then_some("deltaTime"),
        "Stepped" => match param_idx {
            0 => Some("time"),
            1 => Some("deltaTime"),
            _ => None,
        },
        "ChildAdded" | "ChildRemoved" => (param_idx == 0).then_some("child"),
        "DescendantAdded" | "DescendantRemoving" => (param_idx == 0).then_some("descendant"),
        "HealthChanged" => (param_idx == 0).then_some("health"),
        "Chatted" => (param_idx == 0).then_some("message"),
        "FocusLost" => match param_idx {
            0 => Some("enterPressed"),
            1 => Some("inputObject"),
            _ => None,
        },
        "Triggered" => (param_idx == 0).then_some("player"),
        "PromptTriggered" | "PromptButtonHoldBegan" => match param_idx {
            0 => Some("prompt"),
            1 => Some("player"),
            _ => None,
        },
        "Idled" => (param_idx == 0).then_some("idleTime"),
        "MouseButton1Down" | "MouseButton1Up" => match param_idx {
            0 => Some("x"),
            1 => Some("y"),
            _ => None,
        },
        "Activated" => match param_idx {
            0 => Some("inputObject"),
            1 => Some("clickCount"),
            _ => None,
        },
        "AncestryChanged" => match param_idx {
            0 => Some("child"),
            1 => Some("parent"),
            _ => None,
        },
        "AttributeChanged" => (param_idx == 0).then_some("attribute"),
        "StateChanged" => match param_idx {
            0 => Some("old"),
            1 => Some("new"),
            _ => None,
        },
        "MoveToFinished" => (param_idx == 0).then_some("reached"),
        "Completed" => (param_idx == 0).then_some("playbackState"),
        _ => None,
    }
}

/// Converts arbitrary identifiers to idiomatic Luau camelCase.
pub fn to_camel_case(s: &str) -> Option<String> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }

    let words: Vec<&str> = s
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();

    if words.is_empty() {
        return None;
    }

    let mut out = String::with_capacity(32);
    for (i, word) in words.iter().enumerate() {
        if i == 0 {
            let chars: Vec<char> = word.chars().collect();
            if chars.iter().all(|c| c.is_uppercase()) {
                for c in chars {
                    out.extend(c.to_lowercase());
                }
            } else if chars.len() > 1 && chars[0].is_uppercase() && chars[1].is_uppercase() {
                let split_idx = chars
                    .windows(2)
                    .position(|w| w[0].is_uppercase() && w[1].is_lowercase())
                    .unwrap_or(chars.len() - 1);
                for c in &chars[..split_idx] {
                    out.extend(c.to_lowercase());
                }
                for c in &chars[split_idx..] {
                    out.push(*c);
                }
            } else {
                for (ci, c) in chars.iter().enumerate() {
                    if ci == 0 {
                        out.extend(c.to_lowercase());
                    } else {
                        out.push(*c);
                    }
                }
            }
        } else {
            let mut chars = word.chars();
            if let Some(first) = chars.next() {
                out.extend(first.to_uppercase());
                out.extend(chars);
            }
        }
    }

    if out.starts_with(|c: char| c.is_ascii_digit()) {
        out.insert(0, 'n');
    }

    if LUAU_KEYWORDS.contains(&out.as_str()) {
        out.push('_');
    }

    if out.len() > 32 {
        out.truncate(32);
    }

    Some(out)
}

/// Singularizes plural identifiers for loop variables:
/// "children" -> "child", "players" -> "player", "parts" -> "part".
pub fn singularize(name: &str) -> Option<String> {
    if name == "children" {
        return Some("child".to_string());
    }
    if name.ends_with("ies") && name.len() > 4 {
        return Some(format!("{}y", &name[..name.len() - 3]));
    }
    if name.ends_with('s') && !name.ends_with("ss") && name.len() > 3 {
        return Some(name[..name.len() - 1].to_string());
    }
    None
}

/// Roblox Semantic Inference Engine.
pub struct RobloxSemanticInference;

impl RobloxSemanticInference {
    /// Extracts string literal from an IrExpr if present.
    pub fn extract_str<'a, 'arena>(expr: &'a IrExpr<'arena>) -> Option<&'a str> {
        match expr {
            IrExpr::String(s) => Some(*s),
            _ => None,
        }
    }

    /// Infers semantic hint from an assignment RHS expression.
    pub fn infer_from_expr<'arena>(expr: &IrExpr<'arena>, pc: usize, ret_idx: usize) -> Option<SemanticHint> {
        match expr {
            IrExpr::Call { callee, args } => Self::infer_from_call(callee, args, pc, ret_idx),
            IrExpr::TableGet { table, key } => {
                if let IrExpr::Global(gt) = &**table {
                    if let Some(target) = lookup_datatype(gt) {
                        return Some(SemanticHint::new(target, HintPriority::Strong, pc));
                    }
                }
                if let Some(prop) = Self::extract_str(key) {
                    if let Some(camel) = to_camel_case(prop) {
                        return Some(SemanticHint::new(camel, HintPriority::Weak, pc));
                    }
                }
                None
            }
            IrExpr::Global(name) => match *name {
                "workspace" => Some(SemanticHint::new("workspace", HintPriority::Strong, pc)),
                "game" => Some(SemanticHint::new("game", HintPriority::Strong, pc)),
                "script" => Some(SemanticHint::new("script", HintPriority::Strong, pc)),
                _ => None,
            },
            IrExpr::Binary { op: BinOp::Concat, left, right: _ } => {
                if let Some(s) = Self::extract_str(left) {
                    if let Some(idx) = s.find(':').or_else(|| s.find('=')) {
                        let prefix = s[..idx].trim();
                        if let Some(camel) = to_camel_case(prefix) {
                            return Some(SemanticHint::new(camel, HintPriority::Label, pc));
                        }
                    }
                }
                Some(SemanticHint::new("str", HintPriority::Weak, pc))
            }
            IrExpr::Binary { op, .. } => match op {
                BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::And | BinOp::Or => {
                    Some(SemanticHint::new("flag", HintPriority::Weak, pc))
                }
                _ => Some(SemanticHint::new("n", HintPriority::Weak, pc)),
            },
            IrExpr::Unary { op: UnOp::Not, .. } => {
                Some(SemanticHint::new("flag", HintPriority::Weak, pc))
            }
            IrExpr::Boolean(_) => Some(SemanticHint::new("flag", HintPriority::Weak, pc)),
            IrExpr::String(_) => Some(SemanticHint::new("str", HintPriority::Weak, pc)),
            IrExpr::Number(_) => Some(SemanticHint::new("n", HintPriority::Weak, pc)),
            IrExpr::NewTable { .. } => Some(SemanticHint::new("tbl", HintPriority::Weak, pc)),
            IrExpr::Closure { .. } => Some(SemanticHint::new("fn", HintPriority::Weak, pc)),
            IrExpr::Nil => Some(SemanticHint::new("val", HintPriority::Weak, pc)),
            _ => None,
        }
    }

    /// Infers semantic hint from a function call.
    pub fn infer_from_call<'arena>(
        callee: &IrExpr<'arena>,
        args: &[IrExpr<'arena>],
        pc: usize,
        ret_idx: usize,
    ) -> Option<SemanticHint> {
        // Multi-return calls: pcall / xpcall
        if let IrExpr::Global(name) = callee {
            if *name == "pcall" || *name == "xpcall" {
                let hint = if ret_idx == 0 { "ok" } else { "result" };
                return Some(SemanticHint::new(hint, HintPriority::Strong, pc));
            }
            if let Some(hint) = lookup_global(name) {
                return Some(SemanticHint::new(hint, HintPriority::Weak, pc));
            }
            if matches!(*name, "WaitForChild" | "FindFirstChild") {
                let child_arg = args.iter().find_map(Self::extract_str);
                if let Some(child_name) = child_arg {
                    if let Some(camel) = to_camel_case(child_name) {
                        return Some(SemanticHint::new(camel, HintPriority::Strong, pc));
                    }
                }
                return Some(SemanticHint::new("child", HintPriority::Strong, pc));
            }
            if matches!(*name, "InvokeServer" | "FireServer") {
                return Some(SemanticHint::new("response", HintPriority::Strong, pc));
            }
            if matches!(*name, "Connect") {
                return Some(SemanticHint::new("connection", HintPriority::Strong, pc));
            }
        }

        // Method calls and member calls: TableGet { table, key }
        if let IrExpr::TableGet { table, key } = callee {
            // Case 1: Datatype constructor (e.g. Vector3.new)
            if let IrExpr::Global(dt) = &**table {
                if let Some(target) = lookup_datatype(dt) {
                    return Some(SemanticHint::new(target, HintPriority::Strong, pc));
                }
                if *dt == "Instance" {
                    if let Some("new") = Self::extract_str(key) {
                        if let Some(first_arg) = args.first().and_then(Self::extract_str) {
                            if let Some(camel) = to_camel_case(first_arg) {
                                return Some(SemanticHint::new(camel, HintPriority::Strong, pc));
                            }
                        }
                        return Some(SemanticHint::new("instance", HintPriority::Strong, pc));
                    }
                }
                if *dt == "Drawing" {
                    if let Some("new") = Self::extract_str(key) {
                        if let Some(first_arg) = args.first().and_then(Self::extract_str) {
                            if let Some(camel) = to_camel_case(first_arg) {
                                return Some(SemanticHint::new(camel, HintPriority::Strong, pc));
                            }
                        }
                    }
                }
            }

            // Case 2: Object method calls
            if let Some(method) = Self::extract_str(key) {
                // GetService / FindService - Canonical Roblox Services
                if method == "GetService" || method == "FindService" {
                    if let Some(svc) = args.iter().find_map(Self::extract_str) {
                        let name = match svc {
                            "Players" => "players",
                            "ReplicatedStorage" => "replicatedStorage",
                            "HttpService" => "httpService",
                            "UserInputService" => "userInputService",
                            "RunService" => "runService",
                            "Workspace" => "workspaceService",
                            "TweenService" => "tweenService",
                            "Lighting" => "lighting",
                            "CoreGui" => "coreGui",
                            "SoundService" => "soundService",
                            "StarterGui" => "starterGui",
                            "TeleportService" => "teleportService",
                            other => return to_camel_case(other).map(|c| SemanticHint::new(c, HintPriority::Strong, pc)),
                        };
                        return Some(SemanticHint::new(name, HintPriority::Strong, pc));
                    }
                }
                // Child finders (WaitForChild, FindFirstChild, etc.)
                if matches!(
                    method,
                    "WaitForChild"
                        | "FindFirstChild"
                        | "FindFirstChildOfClass"
                        | "FindFirstChildWhichIsA"
                        | "FindFirstAncestor"
                        | "FindFirstAncestorOfClass"
                        | "FindFirstAncestorWhichIsA"
                ) {
                    let child_arg = args.iter().find_map(Self::extract_str);
                    if let Some(child_name) = child_arg {
                        if let Some(camel) = to_camel_case(child_name) {
                            return Some(SemanticHint::new(camel, HintPriority::Strong, pc));
                        }
                    }
                    return Some(SemanticHint::new("child", HintPriority::Strong, pc));
                }
                // Remote invocation & event firing
                if matches!(method, "InvokeServer" | "InvokeClient") {
                    return Some(SemanticHint::new("response", HintPriority::Strong, pc));
                }
                if matches!(method, "FireServer" | "FireClient" | "FireAllClients") {
                    return Some(SemanticHint::new("response", HintPriority::Strong, pc));
                }
                // Signal connections
                if matches!(method, "Connect" | "Once" | "ConnectParallel") {
                    return Some(SemanticHint::new("connection", HintPriority::Strong, pc));
                }
                // IsA check
                if method == "IsA" {
                    if let Some(class_name) = args.iter().find_map(Self::extract_str) {
                        if let Some(camel) = to_camel_case(class_name) {
                            return Some(SemanticHint::new(format!("is{}", camel), HintPriority::Strong, pc));
                        }
                    }
                    return Some(SemanticHint::new("isA", HintPriority::Strong, pc));
                }
                // Standard method lookup
                if let Some(mapped) = lookup_method(method) {
                    return Some(SemanticHint::new(mapped, HintPriority::Strong, pc));
                }
                // Get<X> prefix extraction
                if method.starts_with("Get") && method.len() > 3 {
                    if let Some(camel) = to_camel_case(&method[3..]) {
                        return Some(SemanticHint::new(camel, HintPriority::Strong, pc));
                    }
                }
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bumpalo::Bump;
    use crate::cfg::ir::IrExpr;
    use crate::ssa::{VariableWeb, WebId};
    use crate::namer::{IntervalNamer, NamerConfig};

    #[test]
    fn test_canonical_roblox_services() {
        let bump = Bump::new();
        let game = bump.alloc(IrExpr::Global("game"));
        let get_svc = bump.alloc(IrExpr::String("GetService"));
        let callee = bump.alloc(IrExpr::TableGet { table: game, key: get_svc });

        let services = [
            ("Players", "players"),
            ("ReplicatedStorage", "replicatedStorage"),
            ("HttpService", "httpService"),
            ("UserInputService", "userInputService"),
            ("RunService", "runService"),
            ("Workspace", "workspaceService"),
            ("TweenService", "tweenService"),
            ("Lighting", "lighting"),
            ("CoreGui", "coreGui"),
            ("SoundService", "soundService"),
            ("StarterGui", "starterGui"),
            ("TeleportService", "teleportService"),
        ];

        for (svc_name, expected_ident) in services {
            let args: &[IrExpr] = bump.alloc_slice_clone(&[
                IrExpr::Global("game"),
                IrExpr::String(svc_name),
            ]);
            let call = IrExpr::Call { callee, args };
            let hint = RobloxSemanticInference::infer_from_expr(&call, 1, 0)
                .unwrap_or_else(|| panic!("Service {} must be recognized", svc_name));
            assert_eq!(hint.name, expected_ident, "Service {} inferred name mismatch", svc_name);
            assert_eq!(hint.priority, HintPriority::Strong);
        }
    }

    #[test]
    fn test_common_roblox_methods() {
        let bump = Bump::new();
        let obj = bump.alloc(IrExpr::Register(0));

        // 1. WaitForChild with arg
        let wfc_k = bump.alloc(IrExpr::String("WaitForChild"));
        let wfc_callee = bump.alloc(IrExpr::TableGet { table: obj, key: wfc_k });
        let wfc_args: &[IrExpr] = bump.alloc_slice_clone(&[IrExpr::Register(0), IrExpr::String("Torso")]);
        let wfc_call = IrExpr::Call { callee: wfc_callee, args: wfc_args };
        let hint = RobloxSemanticInference::infer_from_expr(&wfc_call, 1, 0).unwrap();
        assert_eq!(hint.name, "torso");
        assert_eq!(hint.priority, HintPriority::Strong);

        // 2. WaitForChild without string arg -> "child"
        let wfc_dyn_args: &[IrExpr] = bump.alloc_slice_clone(&[IrExpr::Register(0), IrExpr::Register(1)]);
        let wfc_dyn_call = IrExpr::Call { callee: wfc_callee, args: wfc_dyn_args };
        let hint = RobloxSemanticInference::infer_from_expr(&wfc_dyn_call, 2, 0).unwrap();
        assert_eq!(hint.name, "child");

        // 3. FindFirstChild with arg
        let ffc_k = bump.alloc(IrExpr::String("FindFirstChild"));
        let ffc_callee = bump.alloc(IrExpr::TableGet { table: obj, key: ffc_k });
        let ffc_args: &[IrExpr] = bump.alloc_slice_clone(&[IrExpr::Register(0), IrExpr::String("Humanoid")]);
        let ffc_call = IrExpr::Call { callee: ffc_callee, args: ffc_args };
        let hint = RobloxSemanticInference::infer_from_expr(&ffc_call, 3, 0).unwrap();
        assert_eq!(hint.name, "humanoid");

        // 4. FindFirstChild without string arg -> "child"
        let ffc_dyn_args: &[IrExpr] = bump.alloc_slice_clone(&[IrExpr::Register(0)]);
        let ffc_dyn_call = IrExpr::Call { callee: ffc_callee, args: ffc_dyn_args };
        let hint = RobloxSemanticInference::infer_from_expr(&ffc_dyn_call, 4, 0).unwrap();
        assert_eq!(hint.name, "child");

        // 5. FireServer -> "response"
        let fs_k = bump.alloc(IrExpr::String("FireServer"));
        let fs_callee = bump.alloc(IrExpr::TableGet { table: obj, key: fs_k });
        let fs_args: &[IrExpr] = bump.alloc_slice_clone(&[IrExpr::Register(0), IrExpr::Number(42.0)]);
        let fs_call = IrExpr::Call { callee: fs_callee, args: fs_args };
        let hint = RobloxSemanticInference::infer_from_expr(&fs_call, 5, 0).unwrap();
        assert_eq!(hint.name, "response");

        // 6. InvokeServer -> "response"
        let is_k = bump.alloc(IrExpr::String("InvokeServer"));
        let is_callee = bump.alloc(IrExpr::TableGet { table: obj, key: is_k });
        let is_call = IrExpr::Call { callee: is_callee, args: fs_args };
        let hint = RobloxSemanticInference::infer_from_expr(&is_call, 6, 0).unwrap();
        assert_eq!(hint.name, "response");

        // 7. Connect -> "connection"
        let conn_k = bump.alloc(IrExpr::String("Connect"));
        let conn_callee = bump.alloc(IrExpr::TableGet { table: obj, key: conn_k });
        let conn_call = IrExpr::Call { callee: conn_callee, args: fs_args };
        let hint = RobloxSemanticInference::infer_from_expr(&conn_call, 7, 0).unwrap();
        assert_eq!(hint.name, "connection");
    }

    #[test]
    fn test_typed_variable_names_inference() {
        let bump = Bump::new();

        // str
        let s_expr = IrExpr::String("hello");
        let h_str = RobloxSemanticInference::infer_from_expr(&s_expr, 1, 0).unwrap();
        assert_eq!(h_str.name, "str");
        assert_eq!(h_str.priority, HintPriority::Weak);

        // n
        let n_expr = IrExpr::Number(3.14);
        let h_n = RobloxSemanticInference::infer_from_expr(&n_expr, 2, 0).unwrap();
        assert_eq!(h_n.name, "n");
        assert_eq!(h_n.priority, HintPriority::Weak);

        // flag
        let b_expr = IrExpr::Boolean(true);
        let h_flag = RobloxSemanticInference::infer_from_expr(&b_expr, 3, 0).unwrap();
        assert_eq!(h_flag.name, "flag");
        assert_eq!(h_flag.priority, HintPriority::Weak);

        // tbl
        let t_expr = IrExpr::NewTable { array_size: 0, hash_size: 0 };
        let h_tbl = RobloxSemanticInference::infer_from_expr(&t_expr, 4, 0).unwrap();
        assert_eq!(h_tbl.name, "tbl");
        assert_eq!(h_tbl.priority, HintPriority::Weak);

        // fn
        let fn_expr = IrExpr::Closure { proto_id: 1, upvalues: &[] };
        let h_fn = RobloxSemanticInference::infer_from_expr(&fn_expr, 5, 0).unwrap();
        assert_eq!(h_fn.name, "fn");
        assert_eq!(h_fn.priority, HintPriority::Weak);

        // val
        let nil_expr = IrExpr::Nil;
        let h_val = RobloxSemanticInference::infer_from_expr(&nil_expr, 6, 0).unwrap();
        assert_eq!(h_val.name, "val");
        assert_eq!(h_val.priority, HintPriority::Weak);

        // callback
        let obj = bump.alloc(IrExpr::Register(0));
        let cb_k = bump.alloc(IrExpr::String("Callback"));
        let cb_get = IrExpr::TableGet { table: obj, key: cb_k };
        let h_cb = RobloxSemanticInference::infer_from_expr(&cb_get, 7, 0).unwrap();
        assert_eq!(h_cb.name, "callback");
        assert_eq!(h_cb.priority, HintPriority::Weak);
    }

    #[test]
    fn test_weak_hint_threshold_lowered_in_namer() {
        let namer = IntervalNamer::new(NamerConfig::default());

        let mut w1 = VariableWeb::new(WebId::new(0), 0);
        w1.live_interval = (0, 10);
        w1.semantic_hint = Some(SemanticHint::new("str", HintPriority::Weak, 0));

        let mut w2 = VariableWeb::new(WebId::new(1), 1);
        w2.live_interval = (5, 15);
        w2.semantic_hint = Some(SemanticHint::new("str", HintPriority::Weak, 5));

        let mut webs = vec![w1, w2];
        namer.assign_names(&mut webs);

        // Weak hints must now be honored instead of falling back to v0/v1
        assert_eq!(webs[0].assigned_name, "str");
        assert_eq!(webs[1].assigned_name, "str2");
    }
}
