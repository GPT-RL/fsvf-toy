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
      inherit (pkgs) poetry2nix;
      inherit (poetry2nix) mkPoetryEnv;
      poetryArgs = rec {
        overrides = poetry2nix.overrides.withDefaults (pyfinal: pyprev: let
          poetryArgs = old: {
            buildInputs = (old.buildInputs or []) ++ [pyfinal.poetry];
          };
        in {
          dm-tree = python.pkgs.dm-tree.override {
            inherit
              (pyprev)
              stdenv
              absl-py
              attrs
              buildPythonPackage
              numpy
              pybind11
              wrapt
              ;
          };
          jaxlib = pyprev.jaxlib.override {
            inherit
              (pyprev)
              absl-py
              flatbuffers
              numpy
              scipy
              six
              ;
          };
          setuptools-scm = pyprev.buildPythonPackage rec {
            pname = "setuptools_scm";
            version = "7.0.5";
            src = pyprev.fetchPypi {
              inherit pname version;
              sha256 = "sha256-Ax4Tr3cdb4krlBrbbqBFRbv5Hrxc5ox4qvP/9uH7SEQ=";
            };
            buildInputs = with pyprev; [
              packaging
              tomli
              typing-extensions
              pytest
            ];
          };
        });
        projectDir = ./.;
        python = pkgs.python39;
      };
      poetryApp = poetryArgs;
      poetryEnv = mkPoetryEnv poetryArgs;
    in {
      devShell = pkgs.mkShell rec {
        buildInputs = with pkgs; [poetryEnv];
      };
    });
}
