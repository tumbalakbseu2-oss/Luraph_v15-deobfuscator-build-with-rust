//! Unit and integration tests for Luau AST emission and idiom passes (Milestone 5).

use bumpalo::Bump;
use luraph_deobf_rs::codegen::emitter::{is_valid_ident, EmitterConfig, LuauEmitter};
use luraph_deobf_rs::codegen::idioms::{
    coalesce_if_elseif, fold_table_constructors, rewrite_compound_assignments, rewrite_method_calls,
    strip_trailing_returns, IdiomPipeline,
};
use luraph_deobf_rs::codegen::{
    BinaryOp, CompoundOp, LuauExpr, LuauStat, LuauTableEntry, UnaryOp,
};

#[test]
fn test_method_call_sugar() {
    let bump = Bump::new();

    // Case 1: obj.Method(obj, 1, 2) -> obj:Method(1, 2)
    let call_stat = LuauStat::Call(LuauExpr::Call {
        func: Box::new(LuauExpr::Index {
            table: Box::new(LuauExpr::Identifier("obj")),
            key: Box::new(LuauExpr::String("Method")),
        }),
        args: vec![
            LuauExpr::Identifier("obj"),
            LuauExpr::Number(1.0),
            LuauExpr::Number(2.0),
        ],
    });

    let transformed = rewrite_method_calls(vec![call_stat]);
    assert_eq!(transformed.len(), 1);
    match &transformed[0] {
        LuauStat::Call(LuauExpr::MethodCall { target, method, args }) => {
            assert_eq!(**target, LuauExpr::Identifier("obj"));
            assert_eq!(*method, "Method");
            assert_eq!(args.len(), 2);
            assert_eq!(args[0], LuauExpr::Number(1.0));
            assert_eq!(args[1], LuauExpr::Number(2.0));
        }
        other => panic!("Expected MethodCall, got {other:?}"),
    }

    // Case 2: obj.Method(other, 1, 2) -> remains unchanged
    let call_other = LuauStat::Call(LuauExpr::Call {
        func: Box::new(LuauExpr::Index {
            table: Box::new(LuauExpr::Identifier("obj")),
            key: Box::new(LuauExpr::String("Method")),
        }),
        args: vec![
            LuauExpr::Identifier("other"),
            LuauExpr::Number(1.0),
            LuauExpr::Number(2.0),
        ],
    });
    let unchanged = rewrite_method_calls(vec![call_other.clone()]);
    assert_eq!(unchanged, vec![call_other]);

    // Case 3: Pipeline integration
    let pipeline = IdiomPipeline::default();
    let pipe_transformed = pipeline.transform(
        vec![LuauStat::Call(LuauExpr::Call {
            func: Box::new(LuauExpr::Index {
                table: Box::new(LuauExpr::Identifier("workspace")),
                key: Box::new(LuauExpr::String("FindFirstChild")),
            }),
            args: vec![
                LuauExpr::Identifier("workspace"),
                LuauExpr::String("Part"),
            ],
        })],
        &bump,
    );
    assert_eq!(pipe_transformed.len(), 1);
    let mut out = String::new();
    pipe_transformed[0].emit(0, &mut out);
    assert_eq!(out.trim(), "workspace:FindFirstChild(\"Part\")");
}

#[test]
fn test_compound_assignment_recovery() {
    // Case 1: x = x + 1 -> x += 1
    let add_stat = LuauStat::Assign {
        targets: vec![LuauExpr::Identifier("x")],
        values: vec![LuauExpr::Binary {
            op: BinaryOp::Add,
            left: Box::new(LuauExpr::Identifier("x")),
            right: Box::new(LuauExpr::Number(1.0)),
        }],
    };
    let transformed = rewrite_compound_assignments(vec![add_stat]);
    assert_eq!(transformed.len(), 1);
    match &transformed[0] {
        LuauStat::CompoundAssign { target, op, value } => {
            assert_eq!(*target, LuauExpr::Identifier("x"));
            assert_eq!(*op, CompoundOp::Add);
            assert_eq!(*value, LuauExpr::Number(1.0));
        }
        other => panic!("Expected CompoundAssign, got {other:?}"),
    }

    // Case 2: str = str .. "suffix" -> str ..= "suffix"
    let concat_stat = LuauStat::Assign {
        targets: vec![LuauExpr::Identifier("str")],
        values: vec![LuauExpr::Binary {
            op: BinaryOp::Concat,
            left: Box::new(LuauExpr::Identifier("str")),
            right: Box::new(LuauExpr::String("suffix")),
        }],
    };
    let concat_transformed = rewrite_compound_assignments(vec![concat_stat]);
    assert_eq!(
        concat_transformed,
        vec![LuauStat::CompoundAssign {
            target: LuauExpr::Identifier("str"),
            op: CompoundOp::Concat,
            value: LuauExpr::String("suffix"),
        }]
    );

    // Case 3: multi-target assignment a, b = a + 1, b -> must NOT become compound
    let multi_stat = LuauStat::Assign {
        targets: vec![LuauExpr::Identifier("a"), LuauExpr::Identifier("b")],
        values: vec![
            LuauExpr::Binary {
                op: BinaryOp::Add,
                left: Box::new(LuauExpr::Identifier("a")),
                right: Box::new(LuauExpr::Number(1.0)),
            },
            LuauExpr::Identifier("b"),
        ],
    };
    let multi_transformed = rewrite_compound_assignments(vec![multi_stat.clone()]);
    assert_eq!(multi_transformed, vec![multi_stat]);
}

#[test]
fn test_operator_precedence_parentheses() {
    let emitter = LuauEmitter::new(EmitterConfig::default());

    // a + b * c -> "a + b * c" (no parens around b * c)
    let add_mul = LuauExpr::Binary {
        op: BinaryOp::Add,
        left: Box::new(LuauExpr::Identifier("a")),
        right: Box::new(LuauExpr::Binary {
            op: BinaryOp::Mul,
            left: Box::new(LuauExpr::Identifier("b")),
            right: Box::new(LuauExpr::Identifier("c")),
        }),
    };
    assert_eq!(emitter.format_expr(&add_mul, 0), "a + b * c");

    // (a + b) * c -> "(a + b) * c" (parens around a + b required)
    let mul_add = LuauExpr::Binary {
        op: BinaryOp::Mul,
        left: Box::new(LuauExpr::Binary {
            op: BinaryOp::Add,
            left: Box::new(LuauExpr::Identifier("a")),
            right: Box::new(LuauExpr::Identifier("b")),
        }),
        right: Box::new(LuauExpr::Identifier("c")),
    };
    assert_eq!(emitter.format_expr(&mul_add, 0), "(a + b) * c");

    // a - (b - c) -> "a - (b - c)" (subtraction is left-assoc, right child needs parens)
    let sub_sub = LuauExpr::Binary {
        op: BinaryOp::Sub,
        left: Box::new(LuauExpr::Identifier("a")),
        right: Box::new(LuauExpr::Binary {
            op: BinaryOp::Sub,
            left: Box::new(LuauExpr::Identifier("b")),
            right: Box::new(LuauExpr::Identifier("c")),
        }),
    };
    assert_eq!(emitter.format_expr(&sub_sub, 0), "a - (b - c)");

    // (a - b) - c -> "a - b - c" (left child does NOT need parens)
    let sub_sub_left = LuauExpr::Binary {
        op: BinaryOp::Sub,
        left: Box::new(LuauExpr::Binary {
            op: BinaryOp::Sub,
            left: Box::new(LuauExpr::Identifier("a")),
            right: Box::new(LuauExpr::Identifier("b")),
        }),
        right: Box::new(LuauExpr::Identifier("c")),
    };
    assert_eq!(emitter.format_expr(&sub_sub_left, 0), "a - b - c");

    // a ^ (b ^ c) -> "a ^ b ^ c" (exponentiation is right-associative)
    let pow_pow = LuauExpr::Binary {
        op: BinaryOp::Pow,
        left: Box::new(LuauExpr::Identifier("a")),
        right: Box::new(LuauExpr::Binary {
            op: BinaryOp::Pow,
            left: Box::new(LuauExpr::Identifier("b")),
            right: Box::new(LuauExpr::Identifier("c")),
        }),
    };
    assert_eq!(emitter.format_expr(&pow_pow, 0), "a ^ b ^ c");

    // (a ^ b) ^ c -> "(a ^ b) ^ c" (left child of right-assoc op requires parens)
    let pow_pow_left = LuauExpr::Binary {
        op: BinaryOp::Pow,
        left: Box::new(LuauExpr::Binary {
            op: BinaryOp::Pow,
            left: Box::new(LuauExpr::Identifier("a")),
            right: Box::new(LuauExpr::Identifier("b")),
        }),
        right: Box::new(LuauExpr::Identifier("c")),
    };
    assert_eq!(emitter.format_expr(&pow_pow_left, 0), "(a ^ b) ^ c");

    // a or b and c -> "a or b and c" ('and' binds tighter than 'or')
    let or_and = LuauExpr::Binary {
        op: BinaryOp::Or,
        left: Box::new(LuauExpr::Identifier("a")),
        right: Box::new(LuauExpr::Binary {
            op: BinaryOp::And,
            left: Box::new(LuauExpr::Identifier("b")),
            right: Box::new(LuauExpr::Identifier("c")),
        }),
    };
    assert_eq!(emitter.format_expr(&or_and, 0), "a or b and c");

    // (a or b) and c -> "(a or b) and c"
    let and_or = LuauExpr::Binary {
        op: BinaryOp::And,
        left: Box::new(LuauExpr::Binary {
            op: BinaryOp::Or,
            left: Box::new(LuauExpr::Identifier("a")),
            right: Box::new(LuauExpr::Identifier("b")),
        }),
        right: Box::new(LuauExpr::Identifier("c")),
    };
    assert_eq!(emitter.format_expr(&and_or, 0), "(a or b) and c");

    // Safe formatting: unary negate on negative literal -(-5)
    let neg_neg = LuauExpr::Unary {
        op: UnaryOp::Negate,
        expr: Box::new(LuauExpr::Number(-5.0)),
    };
    let formatted_neg = emitter.format_expr(&neg_neg, 0);
    assert_eq!(formatted_neg, "-(-5)");
    assert!(!formatted_neg.contains("--"), "Must never emit Lua comment start '--'");
}

#[test]
fn test_table_constructor_formatting() {
    let emitter = LuauEmitter::new(EmitterConfig::default());

    // Empty table
    let empty = LuauExpr::Table(Vec::new());
    assert_eq!(emitter.format_expr(&empty, 0), "{}");

    // Inline table with Array, Record, General entries
    let table = LuauExpr::Table(vec![
        LuauTableEntry::Array(LuauExpr::Number(1.0)),
        LuauTableEntry::Record("name", LuauExpr::String("icedragon")),
        LuauTableEntry::General(LuauExpr::Identifier("k"), LuauExpr::Boolean(true)),
    ]);
    let formatted = emitter.format_expr(&table, 0);
    assert_eq!(formatted, "{ 1, name = \"icedragon\", [k] = true }");

    // Table folding pass
    let bump = Bump::new();
    let stats = vec![
        LuauStat::LocalAssign {
            names: vec!["t"],
            values: vec![LuauExpr::Table(vec![])],
        },
        LuauStat::Assign {
            targets: vec![LuauExpr::Index {
                table: Box::new(LuauExpr::Identifier("t")),
                key: Box::new(LuauExpr::Number(1.0)),
            }],
            values: vec![LuauExpr::Number(100.0)],
        },
        LuauStat::Assign {
            targets: vec![LuauExpr::Index {
                table: Box::new(LuauExpr::Identifier("t")),
                key: Box::new(LuauExpr::Number(2.0)),
            }],
            values: vec![LuauExpr::Number(200.0)],
        },
        LuauStat::Assign {
            targets: vec![LuauExpr::Index {
                table: Box::new(LuauExpr::Identifier("t")),
                key: Box::new(LuauExpr::String("foo")),
            }],
            values: vec![LuauExpr::String("bar")],
        },
    ];

    let folded = fold_table_constructors(stats, &bump);
    assert_eq!(folded.len(), 1);
    match &folded[0] {
        LuauStat::LocalAssign { names, values } => {
            assert_eq!(names, &["t"]);
            match &values[0] {
                LuauExpr::Table(entries) => {
                    assert_eq!(entries.len(), 3);
                    assert_eq!(entries[0], LuauTableEntry::Array(LuauExpr::Number(100.0)));
                    assert_eq!(entries[1], LuauTableEntry::Array(LuauExpr::Number(200.0)));
                    assert_eq!(
                        entries[2],
                        LuauTableEntry::Record("foo", LuauExpr::String("bar"))
                    );
                }
                other => panic!("Expected LuauExpr::Table, got {other:?}"),
            }
        }
        other => panic!("Expected LocalAssign, got {other:?}"),
    }
}

#[test]
fn test_if_elseif_ladder() {
    // Nested: if c1 then s1 else if c2 then s2 else s3 end end
    let nested_if = LuauStat::If {
        cond: LuauExpr::Identifier("c1"),
        then_branch: vec![LuauStat::Call(LuauExpr::Identifier("s1"))],
        elseif_branches: vec![],
        else_branch: Some(vec![LuauStat::If {
            cond: LuauExpr::Identifier("c2"),
            then_branch: vec![LuauStat::Call(LuauExpr::Identifier("s2"))],
            elseif_branches: vec![],
            else_branch: Some(vec![LuauStat::Call(LuauExpr::Identifier("s3"))]),
        }]),
    };

    let coalesced = coalesce_if_elseif(vec![nested_if]);
    assert_eq!(coalesced.len(), 1);
    match &coalesced[0] {
        LuauStat::If {
            cond,
            then_branch,
            elseif_branches,
            else_branch,
        } => {
            assert_eq!(*cond, LuauExpr::Identifier("c1"));
            assert_eq!(then_branch.len(), 1);
            assert_eq!(elseif_branches.len(), 1);
            assert_eq!(elseif_branches[0].cond, LuauExpr::Identifier("c2"));
            assert_eq!(elseif_branches[0].body.len(), 1);
            assert!(else_branch.is_some());
            assert_eq!(else_branch.as_ref().unwrap().len(), 1);
        }
        other => panic!("Expected flattened If, got {other:?}"),
    }

    let mut out = String::new();
    coalesced[0].emit(0, &mut out);
    assert!(out.contains("if c1 then"));
    assert!(out.contains("elseif c2 then"));
    assert!(out.contains("else"));
    assert!(out.ends_with("end\n"));
}

#[test]
fn test_strip_trailing_returns() {
    let stats = vec![
        LuauStat::Call(LuauExpr::Identifier("foo")),
        LuauStat::Return(vec![]),
    ];
    let stripped = strip_trailing_returns(stats);
    assert_eq!(stripped.len(), 1);
    assert_eq!(stripped[0], LuauStat::Call(LuauExpr::Identifier("foo")));

    // Meaningful return with value is preserved
    let val_return = vec![
        LuauStat::Call(LuauExpr::Identifier("foo")),
        LuauStat::Return(vec![LuauExpr::Number(42.0)]),
    ];
    let preserved = strip_trailing_returns(val_return);
    assert_eq!(preserved.len(), 2);
}

#[test]
fn test_do_block_indentation() {
    let mut emitter = LuauEmitter::new(EmitterConfig::default());
    let do_stat = LuauStat::DoBlock(vec![
        LuauStat::LocalAssign {
            names: vec!["a"],
            values: vec![LuauExpr::Number(1.0)],
        },
        LuauStat::Call(LuauExpr::Identifier("a")),
    ]);

    emitter.emit_stat(&do_stat);
    let expected = "do\n    local a = 1\n    a\nend\n";
    assert_eq!(emitter.buffer, expected);
}

#[test]
fn test_is_valid_ident() {
    assert!(is_valid_ident("valid_name"));
    assert!(is_valid_ident("_private"));
    assert!(is_valid_ident("CamelCase123"));
    assert!(!is_valid_ident("123bad"));
    assert!(!is_valid_ident("has-dash"));
    assert!(!is_valid_ident("has space"));
    assert!(!is_valid_ident(""));
    assert!(!is_valid_ident("function")); // keyword
    assert!(!is_valid_ident("end")); // keyword
    assert!(!is_valid_ident("local")); // keyword
    assert!(!is_valid_ident("continue")); // Luau keyword
}
