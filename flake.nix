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
      pkgs = import nixpkgs {
        inherit system;
        #overlays = [
        #(final: prev: {
        #apache-beam = prev.apache-beam.override {
        #buildInputs = [prev.spark];
        #};
        #})
        #];
      };
      inherit (pkgs) poetry2nix;
      pythonEnv = pkgs.poetry2nix.mkPoetryEnv {
        projectDir = ./.;
      };
    in {
      devShell = pkgs.mkShell {
        buildInputs = with pkgs; [
          python39Packages.apache-beam
          pkgs.spark
          pkgs.jdk11
        ];
      };
    };
  in
    with utils.lib; eachSystem defaultSystems out;
}
