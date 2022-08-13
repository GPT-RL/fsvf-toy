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
        config.allowUnfree = true;
        config.cudaSupport = true;
      };
      inherit (pkgs) poetry2nix;
      inherit (pkgs.cudaPackages) cudatoolkit;
      inherit (pkgs.linuxPackages) nvidia_x11;
      poetryArgs = {
        overrides = poetry2nix.overrides.withDefaults (pyfinal: pyprev: {
          etils = pyprev.etils.overridePythonAttrs (
            old: {
              src = pkgs.fetchFromGitHub {
                owner = "ethanabrooks";
                repo = "etils";
                rev = "main";
                sha256 = "sha256-GCj4EbznWlGvR0Y3NDFvrjFgnBXbMvokxRiTt9MycFI=";
              };
              buildInputs = (old.buildInputs) ++ [pyfinal.poetry];
            }
          );
          jaxlib = pyprev.jaxlibWithCuda;
        });
        projectDir = ./.;
        python = pkgs.python39;
      };
      poetryApp = poetry2nix.mkPoetryApplication poetryArgs;
      poetryEnv = poetry2nix.mkPoetryEnv poetryArgs;
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
          export pythonfaulthandler=1
          export pythonbreakpoint=ipdb.set_trace
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
