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
        config.allowUnfree = useCuda;
        config.cudaSupport = useCuda;
      };
      python = pkgs.python39.override {
        packageOverrides = pyfinal: pyprev: rec {
          args = {
            inherit (pyfinal) buildPythonPackage fetchPypi;
          };
          clu =
            import ./nixfiles/clu.nix (args
              // {inherit pkgs;});
          dollar-lambda = import ./nixfiles/dollar-lambda.nix (args
            // {
              inherit (pyfinal) pytypeclass;
              inherit (pkgs) lib;
            });
          gym = import ./nixfiles/gym.nix {
            inherit
              (pyfinal)
              buildPythonPackage
              cloudpickle
              gym-notices
              numpy
              pytest
              ;
            inherit (pkgs) fetchFromGitHub;
          };
          gym-notices = import ./nixfiles/gym-notices.nix args;
          pytypeclass = import ./nixfiles/pytypeclass.nix args;
          #tensorflow_datasets =
          #import ./nixfiles/tensorflow-datasets.nix (args
          #// {inherit pyfinal pyprev;});
        };
      };
      runtime = p:
        with p; [
          #apache-beam
          #clu
          dollar-lambda
          flax
          gym
          jax
          jaxlibWithCuda
          pyyaml
          tensorflow
          #tensorflow-datasets
        ];
      dev = p:
        with p; [
          ipython
          ipdb
          black
        ];
      all = p: runtime p ++ dev p;
      pythonEnv =
        pkgs.poetry2nix.mkPoetryEnv
        {
          inherit python;
          projectDir = ./.;
          #++ all python.pkgs;
          #shellHook =
          #''
          #export PYTHONFAULTHANDLER=1
          #export PYTHONBREAKPOINT=ipdb.set_trace
          #set -o allexport
          #source .env
          #set +o allexport
          #''
          #+ pkgs.lib.optionalString useCuda ''
          #export CUDA_PATH=${cudatoolkit.lib}
          #export LD_LIBRARY_PATH=${cudatoolkit.lib}/lib:${nvidia_x11}/lib
          #export EXTRA_LDFLAGS="-l/lib -l${nvidia_x11}/lib"
          #export EXTRA_CCFLAGS="-i/usr/include"
          #'';
        };
    in {
      devShell = pkgs.mkShell rec {
        inherit (pkgs.cudaPackages) cudatoolkit;
        inherit (pkgs.linuxPackages) nvidia_x11;
        buildInputs =
          [pythonEnv pkgs.pre-commit]
          ++ pkgs.lib.optionals useCuda [nvidia_x11 cudatoolkit];
        shellHook = ''
          export pythonfaulthandler=1
          export pythonbreakpoint=ipdb.set_trace
          set -o allexport
          source .env
          set +o allexport
        ''
        + pkgs.lib.optionalString useCuda ''
        export CUDA_PATH=${cudatoolkit.lib}
        export LD_LIBRARY_PATH=${cudatoolkit.lib}/lib:${nvidia_x11}/lib
        export EXTRA_LDFLAGS="-l/lib -l${nvidia_x11}/lib"
        export EXTRA_CCFLAGS="-i/usr/include"
        '';
      };
      #packages.default = python.withPackages all;
    });
}
