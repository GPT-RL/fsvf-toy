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
        overrides =
          poetry2nix.overrides.withDefaults
          (pyfinal: pyprev: {
            tensorflow = python.pkgs.tensorflow;
          });
        projectDir = ./.;
        python = pkgs.python39;
      };
      poetryEnv = mkPoetryEnv poetryArgs;
    in {
      devShell = pkgs.mkShell {
        buildInputs = [poetryEnv];
      };
      packages.default = python.withPackages (p: [p.tensorflow]);
    });
}
