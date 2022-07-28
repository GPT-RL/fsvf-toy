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
      python = let
        packageOverrides = pyfinal: pyprev: {
          etils = import ./etils.nix {
            inherit pkgs pyfinal pyprev;
          };
          dm-tree =
            if system == "x86_64-darwin"
            then
              import ./dm-tree.nix {
                inherit pkgs pyfinal pyprev;
              }
            else pyprev.dm-tree;
          tensorflow-datasets = import ./tensorflow-datasets.nix {
            inherit pkgs pyfinal pyprev;
          };
        };
      in
        pkgs.python39.override {
          inherit packageOverrides;
          self = python;
        };
      runtime = p:
        with p; [
          apache-beam
          tensorflow-datasets
          tensorflow
        ];
      dev = p:
        with p; [
          ipython
          ipdb
          black
        ];
      all = p: runtime p ++ dev p;
    in {
      devShell = pkgs.mkShell {
        buildInputs = [pkgs.pre-commit] ++ all python.pkgs;
        shellHook = ''
          export pythonbreakpoint=ipdb.set_trace
        '';
      };
      packages.default = python.withPackages all;
    });
}
