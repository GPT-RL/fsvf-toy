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
      pkgs = import nixpkgs {inherit system;};
      python = pkgs.python39;
      inherit (pkgs) poetry2nix;
      inherit (poetry2nix) mkPoetryApplication mkPoetryEnv;
      poetryArgs = {
        inherit python;
        overrides = poetry2nix.overrides.withDefaults (pyfinal: pyprev: {
          jaxlib = pyprev.jaxlibWithCuda;
          tensorflow = python.pkgs.tensorflow;
        });
        projectDir = ./.;
      };
      poetryEnv = mkPoetryEnv poetryArgs;
    in {
      devShell = pkgs.mkShell {
        buildInputs = [poetryEnv];
      };
      packages.default = python.withPackages (p: [p.tensorflow]);
    });
}
