{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    utils,
  }:
    utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };
      inherit (pkgs) poetry2nix;
      inherit (pkgs.cudaPackages) cudatoolkit;
      inherit (pkgs.linuxPackages) nvidia_x11;
      inherit (poetry2nix) mkPoetryApplication mkPoetryEnv;
      poetryArgs = {
        overrides = poetry2nix.overrides.withDefaults (pyfinal: pyprev: let
          args = old: {
            buildInputs = (old.buildInputs or []) ++ [pyfinal.poetry];
          };
        in {
          jaxlib = pyprev.jaxlibWithCuda;
          dollar-lambda = pyprev.dollar-lambda.overridePythonAttrs args;
          run-logger = pyprev.run-logger.overridePythonAttrs args;
          pytypeclass = pyprev.pytypeclass.overridePythonAttrs args;
        });
        projectDir = ./.;
        python = pkgs.python39;
      };
      poetryApp = mkPoetryApplication poetryArgs;
      poetryEnv = mkPoetryEnv poetryArgs;
    in {
      devShell = pkgs.mkShell rec {
        buildInputs = with pkgs; [
          cudatoolkit
          nvidia_x11
          poetry
          poetryEnv
          pre-commit
        ];
        shellHook = ''
          export PYTHONFAULTHANDLER=1
          export PYTHONBREAKPOINT=ipdb.set_trace
          set -o allexport
          source .env
          set +o allexport
          export CUDA_PATH=${cudatoolkit.lib}
          export LD_LIBRARY_PATH=${cudatoolkit.lib}/lib:${nvidia_x11}/lib
          export EXTRA_LDFLAGS="-l/lib -l${nvidia_x11}/lib"
          export EXTRA_CCFLAGS="-i/usr/include"
        '';
      };
      packages.default = poetryApp;
    });
}
