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
      all-dependencies = p: run-dependencies p ++ dev-dependencies p;
    in rec {
      devShell = pkgs.mkShell {
        buildInputs = all-dependencies (pkgs.python39Packages);
        shellHook = ''
          export pythonbreakpoint=ipdb.set_trace
        '';
      };
      packages.default = pkgs.python39.withPackages all-dependencies;
    });
}
