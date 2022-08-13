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
      };
      python = pkgs.python39;
      inherit (pkgs) poetry2nix;
      poetryApp =
        poetry2nix.mkPoetryApplication
        {
          inherit python;
          projectDir = ./.;
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
          });
        };
    in {
      devShell = pkgs.mkShell rec {
        buildInputs = with pkgs; [poetry pre-commit];
      };
      packages.default = poetryApp;
    });
}
