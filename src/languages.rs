use tree_sitter::Language;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Grammar {
    Rust,
    Go,
    Java,
    Kotlin,
    Python,
    JavaScript,
    TypeScript,
    Toml,
    Xml,
    Json,
}

impl Grammar {
    pub fn for_extension(ext: &str) -> Option<Grammar> {
        Some(match ext {
            "rs" => Grammar::Rust,
            "go" => Grammar::Go,
            "java" => Grammar::Java,
            "kt" => Grammar::Kotlin,
            "py" => Grammar::Python,
            "js" => Grammar::JavaScript,
            "ts" => Grammar::TypeScript,
            "toml" => Grammar::Toml,
            "xml" => Grammar::Xml,
            "json" => Grammar::Json,
            _ => return None,
        })
    }

    pub fn language(self) -> Language {
        match self {
            Grammar::Rust => tree_sitter_rust::LANGUAGE.into(),
            Grammar::Go => tree_sitter_go::LANGUAGE.into(),
            Grammar::Java => tree_sitter_java::LANGUAGE.into(),
            Grammar::Kotlin => tree_sitter_kotlin_ng::LANGUAGE.into(),
            Grammar::Python => tree_sitter_python::LANGUAGE.into(),
            Grammar::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
            Grammar::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            Grammar::Toml => tree_sitter_toml_ng::LANGUAGE.into(),
            Grammar::Xml => tree_sitter_xml::LANGUAGE_XML.into(),
            Grammar::Json => tree_sitter_json::LANGUAGE.into(),
        }
    }

    /// JSON has no identifier-kind nodes at all: object keys are plain
    /// `string` nodes, indistinguishable by kind from string *values*.
    /// Keys are pulled out structurally instead (see `extract::json_keys`).
    pub fn uses_structural_extraction(self) -> bool {
        matches!(self, Grammar::Json)
    }

    /// Node kinds that name or reference something: identifiers, type/field
    /// names, object/table keys, element and attribute names. Deliberately
    /// excludes reserved language keywords, and excludes dotted/composite
    /// wrapper kinds (e.g. Rust's `scoped_identifier`, Kotlin's
    /// `qualified_identifier`) since a full-tree walk already visits their
    /// leaf identifier children directly.
    pub fn identifier_kinds(self) -> &'static [&'static str] {
        match self {
            Grammar::Rust => &[
                "identifier",
                "type_identifier",
                "field_identifier",
                "shorthand_field_identifier",
            ],
            Grammar::Go => &["identifier", "type_identifier", "field_identifier", "package_identifier"],
            Grammar::Java => &["identifier", "type_identifier"],
            Grammar::Kotlin => &["identifier"],
            Grammar::Python => &["identifier"],
            Grammar::JavaScript => &[
                "identifier",
                "property_identifier",
                "shorthand_property_identifier",
                "shorthand_property_identifier_pattern",
                "private_property_identifier",
                "statement_identifier",
            ],
            Grammar::TypeScript => &[
                "identifier",
                "type_identifier",
                "property_identifier",
                "shorthand_property_identifier",
                "shorthand_property_identifier_pattern",
                "private_property_identifier",
                "statement_identifier",
            ],
            Grammar::Toml => &["bare_key", "quoted_key"],
            Grammar::Xml => &["Name"],
            Grammar::Json => &[],
        }
    }
}

/// Extensions handled by the plain word-tokenizer fallback: no tree-sitter
/// grammar is used for these, since prose formats have no "identifier"
/// concept to walk.
pub fn is_fallback_extension(ext: &str) -> bool {
    matches!(ext, "txt" | "md")
}
