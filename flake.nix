{
  description = "Development environment for the multi-agent theorem prover";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # Python environment with required packages
        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          requests
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          name = "theorem-prover-shell";

          buildInputs = [
            pythonEnv
          ];

          shellHook = ''
            echo "------------------------------------------------"
            echo "🤖 Multi-Agent Theorem Prover Shell"
            echo "------------------------------------------------"
          '';
        };
      }
    );
}
