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
      };
      packages.default = poetryApp;
    });
}
