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
      run-dependencies = p:
        with p; [
          apache-beam
          tensorflow-datasets
          tensorflow
        ];
      dev-dependencies = p:
        with p; [
          ipython
          ipdb
          black
        ];
    in rec {
      devShell = pkgs.mkShell {
        buildInputs =
          run-dependencies (pkgs.python39Packages)
          ++ dev-dependencies (pkgs.python39Packages);
      };
      packages.default = pkgs.python39.withPackages run-dependencies;
    });
}
