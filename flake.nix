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
        overlays = [
          (final: prev: {
            python39 = prev.python39.override {
              packageOverrides = pyfinal: pyprev: {
                tensorflow-datasets = pyprev.tensorflow-datasets.overridePythonAttrs (old: rec {
                  version = "4.6.0";
                  src = prev.fetchFromGitHub {
                    owner = "tensorflow";
                    repo = "datasets";
                    rev = "v${version}";
                    sha256 = "sha256-OZpaY/6BMISq5IeDXyuyu5L/yG+DwlFliw4BsipPOLg=";
                  };
                });
              };
            };
          })
        ];
      };
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
    in {
      devShell = pkgs.mkShell {
        buildInputs = with pkgs; [
          python39
          python39.pkgs.tensorflow-datasets
          python39.pkgs.tensorflow
        ];
        #all-dependencies pkgs.python39.pkgs ++
        shellHook = ''
          export pythonbreakpoint=ipdb.set_trace
        '';
      };
      packages.default = pkgs.python39.withPackages all-dependencies;
    });
}
