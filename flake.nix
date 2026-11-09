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

        # The package source: from self.source when the flake is fetched (so
        # `nix run <url>` works without cloning), from the checkout when it
        # is run from a local clone.
        proofsSrc = if builtins.hasAttr "source" self
          then self.source + "/proofs"
          else ./proofs;

        # The sandbox for runs given --tools (proofs/sandbox.py): Linux only.
        # The wrapper below points proofs at this bwrap through
        # PROOFS_BWRAP, so the sandbox works without bubblewrap installed.
        bwrapExport = pkgs.lib.optionalString pkgs.stdenv.isLinux
          "export PROOFS_BWRAP=${pkgs.bubblewrap}/bin/bwrap";

        # Python environment with required packages
        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          requests
        ]);

        # The installed package: the proofs/ module copied into a Python home,
        # plus a wrapper on PATH. The source comes from the flake itself, so
        # `nix run .` (or `nix run <flake-url>`) works without cloning.
        proofs = pkgs.stdenvNoCC.mkDerivation {
          pname = "proofs";
          version = "0.1.0";
          src = proofsSrc;
          dontBuild = true;
          installPhase = ''
            mkdir -p $out/bin $out/lib
            cp -r . $out/lib/proofs
            find $out/lib/proofs -name "__pycache__" -type d -exec rm -rf {} +
            # $out expands at build time; \$PYTHONPATH and \$@ are escaped so
            # the heredoc leaves them for the wrapper's own runtime.
            cat > $out/bin/proofs <<EOF
#!${pkgs.stdenv.shell}
if [ -n "\$PYTHONPATH" ]; then PYTHONPATH="$out/lib:\$PYTHONPATH"; else PYTHONPATH="$out/lib"; fi
export PYTHONPATH
${bwrapExport}
exec ${pythonEnv}/bin/python -m proofs "\$@"
EOF
            chmod +x $out/bin/proofs
          '';
        };
      in
      {
        packages.default = proofs;

        apps.default = {
          type = "app";
          program = "${proofs}/bin/proofs";
          description = "Distributed proof construction, coordinated through git";
        };

        devShells.default = pkgs.mkShell {
          name = "theorem-prover-shell";

          buildInputs = [
            pythonEnv
            proofs
          ] ++ pkgs.lib.optionals pkgs.stdenv.isLinux [ pkgs.bubblewrap ];

          shellHook = ''
            echo "------------------------------------------------"
            echo "       Multi-Agent Theorem Prover Shell"
            echo "------------------------------------------------"
            echo "       The proofs command is on your PATH."
          '';
        };
      }
    );
}
