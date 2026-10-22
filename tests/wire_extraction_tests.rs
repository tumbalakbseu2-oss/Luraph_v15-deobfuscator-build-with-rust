//! Milestone 7 Iteration 2 Wire Extraction, Base85 Decoding & Assembly Tests.

use luraph_deobf_rs::{
    assemble_luau_script, decode_ascii85_lph, extract_lph_ciphertext,
    AssemblyStrategy, LiftedProto,
};

#[test]
fn test_extract_lph_ciphertext_long_brackets() {
    let script = r#"
        return({
            C = [==[LPH)!!Uu5%eKeo;=mBN'D)]==],
            g = function() end
        }):g()(...)
    "#;
    let extracted = extract_lph_ciphertext(script).expect("ciphertext extracted");
    assert!(extracted.starts_with("LPH)!!Uu5"));
    assert!(extracted.ends_with("mBN'D)"));
}

#[test]
fn test_extract_lph_ciphertext_quotes() {
    let script = r#"
        local payload = "LPH)!!Uu5%eKeo;=mBN'D)"
        return payload
    "#;
    let extracted = extract_lph_ciphertext(script).expect("ciphertext extracted");
    assert_eq!(extracted, "LPH)!!Uu5%eKeo;=mBN'D)");
}

#[test]
fn test_decode_ascii85_lph_empty_and_prefix() {
    // Prefix only
    let empty = decode_ascii85_lph(b"LPH)").expect("decoding prefix succeeds");
    assert_eq!(empty.len(), 0);

    // Standard Ascii85: '!!!!!' -> 4 zero bytes
    let zeros = decode_ascii85_lph(b"LPH)!!!!!").expect("decoding succeeds");
    assert_eq!(zeros, vec![0, 0, 0, 0]);

    // 'z' shorthand -> 4 zero bytes
    let z_shorthand = decode_ascii85_lph(b"LPH)z").expect("decoding succeeds");
    assert_eq!(z_shorthand, vec![0, 0, 0, 0]);
}

#[test]
fn test_multi_prototype_assembly_protos_table() {
    let lifted = vec![
        LiftedProto::new(0, "function(...)\n    return _protos[1]() + _protos[2]()\nend\n"),
        LiftedProto::new(1, "function(...)\n    return 10\nend\n"),
        LiftedProto::new(2, "function(...)\n    return 20\nend\n"),
    ];

    let output = assemble_luau_script(&lifted, AssemblyStrategy::TopLevel);
    assert!(output.contains("local _protos = {}"), "Must declare _protos table");
    assert!(output.contains("_protos[1] = function(...)"), "Must assign child 1");
    assert!(output.contains("_protos[2] = function(...)"), "Must assign child 2");
    assert!(output.contains("return function(...)"), "Must return root function at bottom");

    // Verify ordering: _protos declared before children, children before root return
    let pos_protos_decl = output.find("local _protos = {}").unwrap();
    let pos_p1 = output.find("_protos[1]").unwrap();
    let pos_p2 = output.find("_protos[2]").unwrap();
    let pos_return_root = output.find("return function(...)").unwrap();

    assert!(pos_protos_decl < pos_p1);
    assert!(pos_p1 < pos_p2);
    assert!(pos_p2 < pos_return_root);
}

#[test]
fn test_proto0_vortex_lifting() {
    let script = std::fs::read_to_string("../uploads/VortexMain.lua").unwrap_or_else(|_| {
        std::fs::read_to_string("uploads/VortexMain.lua").expect("read VortexMain.lua")
    });
    let extracted = extract_lph_ciphertext(&script).expect("ciphertext extracted");
    let payload = if extracted.starts_with("LPH)") || extracted.starts_with("LPH~") {
        &extracted[4..]
    } else {
        &extracted[3..]
    };
    let decoded = decode_ascii85_lph(payload.as_bytes()).expect("decode payload");
    let deserializer = luraph_deobf_rs::bytecode::BytecodeDeserializer::new();
    let root_proto = deserializer.deserialize(&decoded).expect("deserialize");
    let protos = root_proto.flatten();
    println!("Total protos: {}", protos.len());
    let proto0 = &protos[0];
    println!("Proto 0: id={}, instructions={}, constants={}", proto0.id, proto0.instructions.len(), proto0.constants.len());
    let bump = bumpalo::Bump::new();
    let src = luraph_deobf_rs::cfg::lift_prototype_with_options(proto0, &bump, false);
    assert!(src.len() > 100, "Proto 0 must lift authentic non-empty statements");
    assert!(!luraph_deobf_rs::is_empty_or_noop_proto(&src), "Proto 0 must not be classified as empty or noop");
}



