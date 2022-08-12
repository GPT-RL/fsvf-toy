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
      useCuda = system == "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
        config.cudaSupport = useCuda;
      };
      python = pkgs.python39.override {
        packageOverrides = pyfinal: pyprev: rec {
          args = {
            inherit (pkgs) lib;
            inherit (pyfinal) buildPythonPackage fetchPypi;
          };

          dollar-lambda = import ./nixfiles/dollar-lambda.nix (args
            // {
              inherit (pyfinal) pytypeclass;
            });
          pytypeclass = import ./nixfiles/pytypeclass.nix args;
        };
      };
      runtime = p:
        with p; [
          apache-beam
          dollar-lambda
          flax
          jax
          jaxlibWithCuda
        ];
      dev = p:
        with p; [
          ipython
          ipdb
          black
        ];
      all = p: runtime p ++ dev p;
    in {
      devShell = pkgs.mkShell rec {
        inherit (pkgs.cudaPackages) cudatoolkit;
        inherit (pkgs.linuxPackages) nvidia_x11;
        buildInputs = with pkgs;
          [
            cudatoolkit
            nvidia_x11
            pre-commit
          ]
          ++ all python.pkgs;
        shellHook = ''
          export pythonfaulthandler=1
          export pythonbreakpoint=ipdb.set_trace
          set -o allexport
          source .env
          set +o allexport

            export CUDA_PATH=${cudatoolkit.lib}
            export LD_LIBRARY_PATH=${cudatoolkit.lib}/lib:${nvidia_x11}/lib
            export EXTRA_LDFLAGS="-l/lib -l${nvidia_x11}/lib"
            export EXTRA_CCFLAGS="-i/usr/include"
        '';
      };
      packages.default = python.withPackages all;
    });
}
