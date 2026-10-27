import Lake
open Lake DSL

package ProjReLU where
  srcDir := "Lean"
  leanOptions := #[⟨`autoImplicit, false⟩]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.26.0"

@[default_target]
lean_lib ProjReLU where
  roots := #[`ProjReLU]
