{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    utils,
  }: let
    out = system: let
      pkgs = import nixpkgs {inherit system;};
      inherit (pkgs) poetry2nix;
      pythonEnv = pkgs.poetry2nix.mkPoetryEnv {
        projectDir = ./.;
      };
    in {
      devShell = pkgs.mkShell {
        buildInputs = with pkgs; [
          python39Packages.apache-beam
        ];
      };
    };
  in
    with utils.lib; eachSystem defaultSystems out;
}
