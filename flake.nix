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
          tensorflow-datasets = pyprev.tensorflow-datasets.overridePythonAttrs (old: rec {
            doCheck = false;
            version = "4.6.0";
            src = pyprev.fetchPypi {
              pname = "tensorflow-datasets";
              inherit version;
              sha256 = "sha256-naKmUpfpgfYcXxy0VpYGa5xAE9l4A6bfdDgrwk4hN8s=";
            };
            pythonImportsCheck = null;
            propagatedBuildInputs =
              old.propagatedBuildInputs
              ++ pyfinal.etils.optional-dependencies.epath
              ++ [pyfinal.etils];
          });
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
        buildInputs = all python.pkgs;
        shellHook = ''
          export pythonbreakpoint=ipdb.set_trace
        '';
      };
      packages.default = python.withPackages all;
    });
}
