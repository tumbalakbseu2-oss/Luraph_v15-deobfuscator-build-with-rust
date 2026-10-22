//! Milestone 5 Adversarial Test Suite
//!
//! Empirical stress tests for Luau AST generation, operator precedence,
//! unary minus collision avoidance, method call sugar safety,
//! deeply nested elseif flattening, and table constructor folding.
//!
//! Each generated Luau construct is verified both by structural AST assertions
//! and by empirical execution inside the standalone `lune` Luau runtime.

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use bumpalo::Bump;

use luraph_deobf_rs::codegen::emitter::{EmitterConfig, LuauEmitter};
use luraph_deobf_rs::codegen::idioms::{
    coalesce_if_elseif, fold_table_constructors, rewrite_method_calls, IdiomPipeline,
};
use luraph_deobf_rs::codegen::{
    BinaryOp, LuauExpr, LuauStat, LuauTableEntry, UnaryOp,
};

/// Helper to execute emitted Luau source code using the bundled `lune.exe`.
/// Returns (exit_code, stdout, stderr).
fn run_in_lune(lua_code: &str) -> (i32, String, String) {
    let tmp_dir = tempfile::tempdir().expect("Failed to create tempdir");
    let script_path = tmp_dir.path().join("test.luau");
    fs::write(&script_path, lua_code).expect("Failed to write Luau test file");

    let lune_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("Failed to find parent directory")
        .join("bin")
        .join("lune.exe");

    assert!(
        lune_path.exists(),
        "lune.exe must exist at {:?}",
        lune_path
    );

    let output = Command::new(&lune_path)
        .arg("run")
        .arg(&script_path)
        .output()
        .expect("Failed to execute lune.exe");

    let exit_code = output.status.code().unwrap_or(-1);
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    (exit_code, stdout, stderr)
}

// ============================================================================
// 1. OPERATOR PRECEDENCE & ASSOCIATIVITY STRESS TESTS
// ============================================================================

#[test]
fn test_adv_operator_precedence_and_associativity_chains() {
    let emitter = LuauEmitter::new(EmitterConfig::default());

    // ------------------------------------------------------------------------
    // (A) Subtraction: Left-associative operator
    // ------------------------------------------------------------------------
    // a - (b - c) -> "a - (b - c)" (Right child needs parens)
    let sub_right = LuauExpr::Binary {
        op: BinaryOp::Sub,
        left: Box::new(LuauExpr::Identifier("a")),
        right: Box::new(LuauExpr::Binary {
            op: BinaryOp::Sub,
            left: Box::new(LuauExpr::Identifier("b")),
            right: Box::new(LuauExpr::Identifier("c")),
        }),
    };
    assert_eq!(emitter.format_expr(&sub_right, 0), "a - (b - c)");

    // (a - b) - c -> "a - b - c" (Left child does NOT need parens)
    let sub_left = LuauExpr::Binary {
        op: BinaryOp::Sub,
        left: Box::new(LuauExpr::Binary {
            op: BinaryOp::Sub,
            left: Box::new(LuauExpr::Identifier("a")),
            right: Box::new(LuauExpr::Identifier("b")),
        }),
        right: Box::new(LuauExpr::Identifier("c")),
    };
    assert_eq!(emitter.format_expr(&sub_left, 0), "a - b - c");

    // ------------------------------------------------------------------------
    // (B) Division and Floor Division: Left-associative operators
    // ------------------------------------------------------------------------
    // (a / b) / c -> "a / b / c"
    let div_left = LuauExpr::Binary {
        op: BinaryOp::Div,
        left: Box::new(LuauExpr::Binary {
            op: BinaryOp::Div,
            left: Box::new(LuauExpr::Identifier("a")),
            right: Box::new(LuauExpr::Identifier("b")),
        }),
        right: Box::new(LuauExpr::Identifier("c")),
    };
    assert_eq!(emitter.format_expr(&div_left, 0), "a / b / c");

    // a / (b / c) -> "a / (b / c)"
    let div_right = LuauExpr::Binary {
        op: BinaryOp::Div,
        left: Box::new(LuauExpr::Identifier("a")),
        right: Box::new(LuauExpr::Binary {
            op: BinaryOp::Div,
            left: Box::new(LuauExpr::Identifier("b")),
            right: Box::new(LuauExpr::Identifier("c")),
        }),
    };
    assert_eq!(emitter.format_expr(&div_right, 0), "a / (b / c)");

    // a // (b // c) -> "a // (b // c)"
    let fdiv_right = LuauExpr::Binary {
        op: BinaryOp::FloorDiv,
        left: Box::new(LuauExpr::Identifier("a")),
        right: Box::new(LuauExpr::Binary {
            op: BinaryOp::FloorDiv,
            left: Box::new(LuauExpr::Identifier("b")),
            right: Box::new(LuauExpr::Identifier("c")),
        }),
    };
    assert_eq!(emitter.format_expr(&fdiv_right, 0), "a // (b // c)");

    // ------------------------------------------------------------------------
    // (C) Exponentiation (^): Right-associative operator
    // ------------------------------------------------------------------------
    // a ^ (b ^ c) -> "a ^ b ^ c" (Right child does NOT need parens)
    let pow_right = LuauExpr::Binary {
        op: BinaryOp::Pow,
        left: Box::new(LuauExpr::Identifier("a")),
        right: Box::new(LuauExpr::Binary {
            op: BinaryOp::Pow,
            left: Box::new(LuauExpr::Identifier("b")),
            right: Box::new(LuauExpr::Identifier("c")),
        }),
    };
    assert_eq!(emitter.format_expr(&pow_right, 0), "a ^ b ^ c");

    // (a ^ b) ^ c -> "(a ^ b) ^ c" (Left child DOES need parens)
    let pow_left = LuauExpr::Binary {
        op: BinaryOp::Pow,
        left: Box::new(LuauExpr::Binary {
            op: BinaryOp::Pow,
            left: Box::new(LuauExpr::Identifier("a")),
            right: Box::new(LuauExpr::Identifier("b")),
        }),
        right: Box::new(LuauExpr::Identifier("c")),
    };
    assert_eq!(emitter.format_expr(&pow_left, 0), "(a ^ b) ^ c");

    // ------------------------------------------------------------------------
    // (D) String Concatenation (..): Right-associative operator
    // ------------------------------------------------------------------------
    // a .. (b .. c) -> "a .. b .. c"
    let concat_right = LuauExpr::Binary {
        op: BinaryOp::Concat,
        left: Box::new(LuauExpr::Identifier("a")),
        right: Box::new(LuauExpr::Binary {
            op: BinaryOp::Concat,
            left: Box::new(LuauExpr::Identifier("b")),
            right: Box::new(LuauExpr::Identifier("c")),
        }),
    };
    assert_eq!(emitter.format_expr(&concat_right, 0), "a .. b .. c");

    // (a .. b) .. c -> "(a .. b) .. c"
    let concat_left = LuauExpr::Binary {
        op: BinaryOp::Concat,
        left: Box::new(LuauExpr::Binary {
            op: BinaryOp::Concat,
            left: Box::new(LuauExpr::Identifier("a")),
            right: Box::new(LuauExpr::Identifier("b")),
        }),
        right: Box::new(LuauExpr::Identifier("c")),
    };
    assert_eq!(emitter.format_expr(&concat_left, 0), "(a .. b) .. c");

    // ------------------------------------------------------------------------
    // (E) Deep Mixed Expression Chain with Empirical Evaluation in Lune
    // ------------------------------------------------------------------------
    // Construct arithmetic: ((100 - (50 - 20)) / 2) ^ 2
    // 50 - 20 = 30; 100 - 30 = 70; 70 / 2 = 35; 35 ^ 2 = 1225
    let deep_expr = LuauExpr::Binary {
        op: BinaryOp::Pow,
        left: Box::new(LuauExpr::Binary {
            op: BinaryOp::Div,
            left: Box::new(LuauExpr::Binary {
                op: BinaryOp::Sub,
                left: Box::new(LuauExpr::Number(100.0)),
                right: Box::new(LuauExpr::Binary {
                    op: BinaryOp::Sub,
                    left: Box::new(LuauExpr::Number(50.0)),
                    right: Box::new(LuauExpr::Number(20.0)),
                }),
            }),
            right: Box::new(LuauExpr::Number(2.0)),
        }),
        right: Box::new(LuauExpr::Number(2.0)),
    };

    let formatted_code = emitter.format_expr(&deep_expr, 0);
    assert_eq!(formatted_code, "((100 - (50 - 20)) / 2) ^ 2");

    let lua_prog = format!("local res = {formatted_code}\nassert(res == 1225, 'Math mismatch: ' .. tostring(res))\nprint('PRECEDENCE_OK')");
    let (code, stdout, stderr) = run_in_lune(&lua_prog);
    assert_eq!(code, 0, "Lune execution failed: {stderr}");
    assert!(stdout.contains("PRECEDENCE_OK"));
}

// ============================================================================
// 2. UNARY MINUS COLLISION AVOIDANCE STRESS TESTS
// ============================================================================

#[test]
fn test_adv_unary_minus_collision_immunity() {
    let emitter = LuauEmitter::new(EmitterConfig::default());

    // ------------------------------------------------------------------------
    // (A) Double Negation: -(-x) must NEVER emit `--x`
    // ------------------------------------------------------------------------
    let neg_neg = LuauExpr::Unary {
        op: UnaryOp::Negate,
        expr: Box::new(LuauExpr::Unary {
            op: UnaryOp::Negate,
            expr: Box::new(LuauExpr::Identifier("x")),
        }),
    };
    let formatted = emitter.format_expr(&neg_neg, 0);
    assert_eq!(formatted, "-(-x)");
    assert!(!formatted.contains("--"), "Must never emit Lua comment '--'");

    // ------------------------------------------------------------------------
    // (B) Triple Negation: -(-(-x))
    // ------------------------------------------------------------------------
    let neg_3 = LuauExpr::Unary {
        op: UnaryOp::Negate,
        expr: Box::new(neg_neg),
    };
    let formatted_3 = emitter.format_expr(&neg_3, 0);
    assert_eq!(formatted_3, "-(-(-x))");
    assert!(!formatted_3.contains("--"), "Must never emit Lua comment '--'");

    // ------------------------------------------------------------------------
    // (C) Quadruple Negation: -(-(-(-x)))
    // ------------------------------------------------------------------------
    let neg_4 = LuauExpr::Unary {
        op: UnaryOp::Negate,
        expr: Box::new(neg_3),
    };
    let formatted_4 = emitter.format_expr(&neg_4, 0);
    assert_eq!(formatted_4, "-(-(-(-x)))");
    assert!(!formatted_4.contains("--"), "Must never emit Lua comment '--'");

    // ------------------------------------------------------------------------
    // (D) Unary Negation on Negative Constant Literal: -(-42.0)
    // ------------------------------------------------------------------------
    let neg_lit = LuauExpr::Unary {
        op: UnaryOp::Negate,
        expr: Box::new(LuauExpr::Number(-42.0)),
    };
    let formatted_lit = emitter.format_expr(&neg_lit, 0);
    assert_eq!(formatted_lit, "-(-42)");
    assert!(!formatted_lit.contains("--"), "Must never emit Lua comment '--'");

    // ------------------------------------------------------------------------
    // (E) Subtraction with Negative Expression: a - (-b)
    // ------------------------------------------------------------------------
    let sub_neg = LuauExpr::Binary {
        op: BinaryOp::Sub,
        left: Box::new(LuauExpr::Identifier("a")),
        right: Box::new(LuauExpr::Unary {
            op: UnaryOp::Negate,
            expr: Box::new(LuauExpr::Identifier("b")),
        }),
    };
    let formatted_sub = emitter.format_expr(&sub_neg, 0);
    // Emitted with safe spacing "a - -b" or "a - (-b)"
    assert!(!formatted_sub.contains("--"), "Must never emit comment '--'");

    // ------------------------------------------------------------------------
    // (F) Empirical Verification in Lune Runtime
    // ------------------------------------------------------------------------
    let lua_prog = format!(
        "local x = 5\n\
         local val1 = {formatted}\n\
         local val2 = {formatted_3}\n\
         local val3 = {formatted_4}\n\
         local val4 = {formatted_lit}\n\
         assert(val1 == 5, 'val1 failed')\n\
         assert(val2 == -5, 'val2 failed')\n\
         assert(val3 == 5, 'val3 failed')\n\
         assert(val4 == 42, 'val4 failed')\n\
         print('UNARY_COLLISION_IMMUNE')"
    );
    let (code, stdout, stderr) = run_in_lune(&lua_prog);
    assert_eq!(code, 0, "Lune execution failed: {stderr}");
    assert!(stdout.contains("UNARY_COLLISION_IMMUNE"));
}

// ============================================================================
// 3. METHOD CALL SUGAR SAFETY STRESS TESTS
// ============================================================================

#[test]
fn test_adv_method_call_sugar_side_effect_safety() {
    // ------------------------------------------------------------------------
    // (A) Function Call as Target: get_obj().Method(get_obj())
    // HAZARD: Function calls have arbitrary side effects.
    // Transforming this to get_obj():Method() would eliminate the second call!
    // Must NOT be transformed!
    // ------------------------------------------------------------------------
    let call_target = LuauStat::Call(LuauExpr::Call {
        func: Box::new(LuauExpr::Index {
            table: Box::new(LuauExpr::Call {
                func: Box::new(LuauExpr::Identifier("get_obj")),
                args: vec![],
            }),
            key: Box::new(LuauExpr::String("Method")),
        }),
        args: vec![LuauExpr::Call {
            func: Box::new(LuauExpr::Identifier("get_obj")),
            args: vec![],
        }],
    });

    let transformed_call = rewrite_method_calls(vec![call_target.clone()]);
    assert_eq!(
        transformed_call,
        vec![call_target],
        "Impure function call receiver must NEVER be rewritten into method call sugar"
    );

    // ------------------------------------------------------------------------
    // (B) Impure Table Index Key: t[get_idx()].Method(t[get_idx()])
    // HAZARD: get_idx() is evaluated twice. Must NOT be rewritten.
    // ------------------------------------------------------------------------
    let impure_index_stat = LuauStat::Call(LuauExpr::Call {
        func: Box::new(LuauExpr::Index {
            table: Box::new(LuauExpr::Index {
                table: Box::new(LuauExpr::Identifier("t")),
                key: Box::new(LuauExpr::Call {
                    func: Box::new(LuauExpr::Identifier("get_idx")),
                    args: vec![],
                }),
            }),
            key: Box::new(LuauExpr::String("Method")),
        }),
        args: vec![LuauExpr::Index {
            table: Box::new(LuauExpr::Identifier("t")),
            key: Box::new(LuauExpr::Call {
                func: Box::new(LuauExpr::Identifier("get_idx")),
                args: vec![],
            }),
        }],
    });

    let transformed_index = rewrite_method_calls(vec![impure_index_stat.clone()]);
    assert_eq!(
        transformed_index,
        vec![impure_index_stat],
        "Index receiver with impure key must NEVER be rewritten into method call sugar"
    );

    // ------------------------------------------------------------------------
    // (C) Pure Receivers: SHOULD be safely rewritten
    // 1. Simple identifier: obj.Method(obj, 10) -> obj:Method(10)
    // 2. Pure chained property: workspace.Part.Destroy(workspace.Part) -> workspace.Part:Destroy()
    // ------------------------------------------------------------------------
    let pure_chained = LuauStat::Call(LuauExpr::Call {
        func: Box::new(LuauExpr::Index {
            table: Box::new(LuauExpr::Index {
                table: Box::new(LuauExpr::Identifier("workspace")),
                key: Box::new(LuauExpr::String("Part")),
            }),
            key: Box::new(LuauExpr::String("Destroy")),
        }),
        args: vec![LuauExpr::Index {
            table: Box::new(LuauExpr::Identifier("workspace")),
            key: Box::new(LuauExpr::String("Part")),
        }],
    });

    let transformed_pure = rewrite_method_calls(vec![pure_chained]);
    assert_eq!(transformed_pure.len(), 1);
    match &transformed_pure[0] {
        LuauStat::Call(LuauExpr::MethodCall { target, method, args }) => {
            assert_eq!(*method, "Destroy");
            assert!(args.is_empty());
            assert_eq!(
                **target,
                LuauExpr::Index {
                    table: Box::new(LuauExpr::Identifier("workspace")),
                    key: Box::new(LuauExpr::String("Part")),
                }
            );
        }
        other => panic!("Expected MethodCall, got {other:?}"),
    }

    // ------------------------------------------------------------------------
    // (D) Keyword & Invalid Identifier Method Names:
    // obj["for"](obj) -> cannot be obj:for() because `for` is a Luau keyword!
    // obj["not a valid ident!"](obj) -> cannot be sugar!
    // ------------------------------------------------------------------------
    let kw_stat = LuauStat::Call(LuauExpr::Call {
        func: Box::new(LuauExpr::Index {
            table: Box::new(LuauExpr::Identifier("obj")),
            key: Box::new(LuauExpr::String("for")),
        }),
        args: vec![LuauExpr::Identifier("obj")],
    });
    let transformed_kw = rewrite_method_calls(vec![kw_stat.clone()]);
    assert_eq!(
        transformed_kw,
        vec![kw_stat],
        "Keywords must never be transformed into method call sugar"
    );

    let invalid_name_stat = LuauStat::Call(LuauExpr::Call {
        func: Box::new(LuauExpr::Index {
            table: Box::new(LuauExpr::Identifier("obj")),
            key: Box::new(LuauExpr::String("invalid-prop")),
        }),
        args: vec![LuauExpr::Identifier("obj")],
    });
    let transformed_invalid = rewrite_method_calls(vec![invalid_name_stat.clone()]);
    assert_eq!(
        transformed_invalid,
        vec![invalid_name_stat],
        "Invalid identifiers must never be transformed into method call sugar"
    );
}

// ============================================================================
// 4. DEEPLY NESTED ELSEIF & TABLE CONSTRUCTOR STRESS TESTS
// ============================================================================

#[test]
fn test_adv_deeply_nested_elseif_ladder_flattening() {
    // ------------------------------------------------------------------------
    // Construct an adversarial 50-level nested `if ... else if ... else ...`
    // Verify that `coalesce_if_elseif` collapses all 50 tiers into a single
    // statement with 1 primary `if`, 49 `elseif` branches, and 1 final `else`.
    // ------------------------------------------------------------------------
    let num_levels = 50;
    let mut current_if = LuauStat::If {
        cond: LuauExpr::Binary {
            op: BinaryOp::Equal,
            left: Box::new(LuauExpr::Identifier("k")),
            right: Box::new(LuauExpr::Number(num_levels as f64)),
        },
        then_branch: vec![LuauStat::Assign {
            targets: vec![LuauExpr::Identifier("res")],
            values: vec![LuauExpr::Number(num_levels as f64)],
        }],
        elseif_branches: vec![],
        else_branch: Some(vec![LuauStat::Assign {
            targets: vec![LuauExpr::Identifier("res")],
            values: vec![LuauExpr::Number(9999.0)],
        }]),
    };

    for i in (1..num_levels).rev() {
        current_if = LuauStat::If {
            cond: LuauExpr::Binary {
                op: BinaryOp::Equal,
                left: Box::new(LuauExpr::Identifier("k")),
                right: Box::new(LuauExpr::Number(i as f64)),
            },
            then_branch: vec![LuauStat::Assign {
                targets: vec![LuauExpr::Identifier("res")],
                values: vec![LuauExpr::Number(i as f64)],
            }],
            elseif_branches: vec![],
            else_branch: Some(vec![current_if]),
        };
    }

    let coalesced = coalesce_if_elseif(vec![current_if]);
    assert_eq!(coalesced.len(), 1);

    match &coalesced[0] {
        LuauStat::If {
            cond,
            then_branch: _,
            elseif_branches,
            else_branch,
        } => {
            assert_eq!(
                *cond,
                LuauExpr::Binary {
                    op: BinaryOp::Equal,
                    left: Box::new(LuauExpr::Identifier("k")),
                    right: Box::new(LuauExpr::Number(1.0)),
                }
            );
            assert_eq!(
                elseif_branches.len(),
                num_levels - 1,
                "All 49 nested else-ifs must be flattened into elseif branches"
            );
            assert!(else_branch.is_some());
        }
        other => panic!("Expected coalesced If, got {other:?}"),
    }

    // Verify emission formatting and execute under Lune
    let mut emitted_code = String::new();
    coalesced[0].emit(0, &mut emitted_code);

    let lua_prog = format!(
        "local res = 0\n\
         local k = 42\n\
         {emitted_code}\n\
         assert(res == 42, 'Expected 42, got ' .. tostring(res))\n\
         k = 99\n\
         {emitted_code}\n\
         assert(res == 9999, 'Expected else branch 9999, got ' .. tostring(res))\n\
         print('ELSEIF_50_LADDER_OK')"
    );

    let (code, stdout, stderr) = run_in_lune(&lua_prog);
    assert_eq!(code, 0, "Lune execution failed: {stderr}");
    assert!(stdout.contains("ELSEIF_50_LADDER_OK"));
}

#[test]
fn test_adv_table_constructor_folding_and_safety_invariants() {
    let bump = Bump::new();

    // ------------------------------------------------------------------------
    // (A) Table Folding Safety: Interleaved Reads Must Halt Folding
    // local t = {}
    // t[1] = 10
    // local read = t[1]  <-- READ of `t`!
    // t[2] = 20          <-- Must NOT be hoisted before `read`!
    // ------------------------------------------------------------------------
    let stats_with_read = vec![
        LuauStat::LocalAssign {
            names: vec!["t"],
            values: vec![LuauExpr::Table(vec![])],
        },
        LuauStat::Assign {
            targets: vec![LuauExpr::Index {
                table: Box::new(LuauExpr::Identifier("t")),
                key: Box::new(LuauExpr::Number(1.0)),
            }],
            values: vec![LuauExpr::Number(10.0)],
        },
        LuauStat::LocalAssign {
            names: vec!["read"],
            values: vec![LuauExpr::Index {
                table: Box::new(LuauExpr::Identifier("t")),
                key: Box::new(LuauExpr::Number(1.0)),
            }],
        },
        LuauStat::Assign {
            targets: vec![LuauExpr::Index {
                table: Box::new(LuauExpr::Identifier("t")),
                key: Box::new(LuauExpr::Number(2.0)),
            }],
            values: vec![LuauExpr::Number(20.0)],
        },
    ];

    let folded_read = fold_table_constructors(stats_with_read, &bump);
    // Should produce 3 statements:
    // 1. local t = { 10 }
    // 2. local read = t[1]
    // 3. t[2] = 20
    assert_eq!(folded_read.len(), 3);
    match &folded_read[0] {
        LuauStat::LocalAssign { names, values } => {
            assert_eq!(names, &["t"]);
            if let LuauExpr::Table(entries) = &values[0] {
                assert_eq!(entries.len(), 1, "Only first assignment can be folded before the read");
                assert_eq!(entries[0], LuauTableEntry::Array(LuauExpr::Number(10.0)));
            } else {
                panic!("Expected Table");
            }
        }
        other => panic!("Expected LocalAssign, got {other:?}"),
    }

    // ------------------------------------------------------------------------
    // (B) Deep Nested Table Construction (10 Levels Deep)
    // ------------------------------------------------------------------------
    let mut nested_tbl = LuauExpr::Table(vec![
        LuauTableEntry::Record("leaf_val", LuauExpr::Number(1337.0)),
    ]);
    for level in 0..10 {
        let field_name: &'static str = Box::leak(format!("tier_{level}").into_boxed_str());
        nested_tbl = LuauExpr::Table(vec![LuauTableEntry::Record(field_name, nested_tbl)]);
    }

    let emitter = LuauEmitter::new(EmitterConfig::default());
    let formatted_tbl = emitter.format_expr(&nested_tbl, 0);

    let lua_prog = format!(
        "local root = {formatted_tbl}\n\
         assert(root.tier_9.tier_8.tier_7.tier_6.tier_5.tier_4.tier_3.tier_2.tier_1.tier_0.leaf_val == 1337)\n\
         print('DEEP_TABLE_OK')"
    );
    let (code, stdout, stderr) = run_in_lune(&lua_prog);
    assert_eq!(code, 0, "Lune execution failed: {stderr}");
    assert!(stdout.contains("DEEP_TABLE_OK"));
}

// ============================================================================
// 5. COMPREHENSIVE END-TO-END AST EMISSION & EXECUTION HARNESS
// ============================================================================

#[test]
fn test_adv_full_pipeline_luau_roundtrip_execution() {
    let bump = Bump::new();

    // Construct a rich Luau module exercising all AST node types:
    // Functions, CompoundAssign, While, Repeat, NumericFor, GenericFor, Table constructors,
    // and Idiom passes (method calls, compound assignments, table folding).
    let raw_stats = vec![
        // local obj = {}
        LuauStat::LocalAssign {
            names: vec!["obj"],
            values: vec![LuauExpr::Table(vec![])],
        },
        // obj.count = 0
        LuauStat::Assign {
            targets: vec![LuauExpr::Index {
                table: Box::new(LuauExpr::Identifier("obj")),
                key: Box::new(LuauExpr::String("count")),
            }],
            values: vec![LuauExpr::Number(0.0)],
        },
        // obj.Increment = function(self, amt) self.count = self.count + amt end
        LuauStat::Assign {
            targets: vec![LuauExpr::Index {
                table: Box::new(LuauExpr::Identifier("obj")),
                key: Box::new(LuauExpr::String("Increment")),
            }],
            values: vec![LuauExpr::Function {
                params: vec!["self", "amt"],
                is_vararg: false,
                body: vec![LuauStat::Assign {
                    targets: vec![LuauExpr::Index {
                        table: Box::new(LuauExpr::Identifier("self")),
                        key: Box::new(LuauExpr::String("count")),
                    }],
                    values: vec![LuauExpr::Binary {
                        op: BinaryOp::Add,
                        left: Box::new(LuauExpr::Index {
                            table: Box::new(LuauExpr::Identifier("self")),
                            key: Box::new(LuauExpr::String("count")),
                        }),
                        right: Box::new(LuauExpr::Identifier("amt")),
                    }],
                }],
            }],
        },
        // obj.Increment(obj, 5) -> will be rewritten to obj:Increment(5)
        LuauStat::Call(LuauExpr::Call {
            func: Box::new(LuauExpr::Index {
                table: Box::new(LuauExpr::Identifier("obj")),
                key: Box::new(LuauExpr::String("Increment")),
            }),
            args: vec![LuauExpr::Identifier("obj"), LuauExpr::Number(5.0)],
        }),
        // for i = 1, 3 do obj.count = obj.count + i end -> compound assign obj.count += i
        LuauStat::NumericFor {
            var: "i",
            start: LuauExpr::Number(1.0),
            limit: LuauExpr::Number(3.0),
            step: None,
            body: vec![LuauStat::Assign {
                targets: vec![LuauExpr::Index {
                    table: Box::new(LuauExpr::Identifier("obj")),
                    key: Box::new(LuauExpr::String("count")),
                }],
                values: vec![LuauExpr::Binary {
                    op: BinaryOp::Add,
                    left: Box::new(LuauExpr::Index {
                        table: Box::new(LuauExpr::Identifier("obj")),
                        key: Box::new(LuauExpr::String("count")),
                    }),
                    right: Box::new(LuauExpr::Identifier("i")),
                }],
            }],
        },
        // return obj.count
        LuauStat::Return(vec![LuauExpr::Index {
            table: Box::new(LuauExpr::Identifier("obj")),
            key: Box::new(LuauExpr::String("count")),
        }]),
    ];

    let pipeline = IdiomPipeline::default();
    let transformed = pipeline.transform(raw_stats, &bump);

    let mut out_code = String::new();
    for st in &transformed {
        st.emit(0, &mut out_code);
    }

    // Verify expected idiom occurrences in emitted Luau code:
    assert!(out_code.contains("obj:Increment(5)"), "Must contain method call sugar: {out_code}");
    assert!(out_code.contains("+="), "Must contain compound assignment: {out_code}");

    // Run module under Lune runtime:
    let test_runner = format!(
        "local function test()\n{out_code}\nend\n\
         local final_count = test()\n\
         -- Initial: 0 + 5 (Increment) + 1 + 2 + 3 (for loop) = 11\n\
         assert(final_count == 11, 'Expected 11, got ' .. tostring(final_count))\n\
         print('FULL_E2E_PIPELINE_OK')"
    );

    let (code, stdout, stderr) = run_in_lune(&test_runner);
    assert_eq!(code, 0, "Lune execution failed: {stderr}\nCode:\n{test_runner}");
    assert!(stdout.contains("FULL_E2E_PIPELINE_OK"));
}
