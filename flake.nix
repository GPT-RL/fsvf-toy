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
        inherit (pkgs.cudaPackages) cudatoolkit;
        inherit (pkgs.linuxPackages) nvidia_x11;
      };
      python = pkgs.python39.override {
        packageOverrides = pyfinal: pyprev: {};
        self = python;
      };
      runtime = p:
        with p; [
          apache-beam
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
      devShell = pkgs.mkShell {
        buildInputs = [pkgs.pre-commit] ++ all python.pkgs;
        shellHook = ''
          export pythonfaulthandler=1
          export pythonbreakpoint=ipdb.set_trace
          set -o allexport
          source .env
          set +o allexport
        '';
      };
      packages.default = python.withPackages all;
    });
}
