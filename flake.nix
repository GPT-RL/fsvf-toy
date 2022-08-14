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
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };
      inherit (pkgs) poetry2nix;
      inherit (poetry2nix) mkPoetryApplication mkPoetryEnv;
      inherit (pkgs.cudaPackages) cudatoolkit;
      inherit (pkgs.linuxPackages) nvidia_x11;
      python = pkgs.python39;
      poetryArgs = {
        inherit python;
        projectDir = ./.;
        preferWheels = true;
        overrides =
          poetry2nix.overrides.withDefaults
          (pyfinal: pyprev: rec {
            tensorflow-gpu =
              # Override the nixpkgs bin version instead of
              # poetry2nix version so that rpath is set correctly.
              pyprev.tensorflow-bin.overridePythonAttrs
              (old: {inherit (pyprev.tensorflow-gpu) src version;});
            astunparse = pyprev.astunparse.overridePythonAttrs (old: {
              buildInputs = (old.buildInputs or []) ++ [pyfinal.wheel];
            });
            # Use cuda-enabled jaxlib as required
            jaxlib = pyprev.jaxlibWithCuda.override {
              inherit
                (pyprev)
                absl-py
                flatbuffers
                numpy
                scipy
                six
                ;
            };
          });
      };
      poetryApp = mkPoetryApplication poetryArgs;
      poetryEnv = mkPoetryEnv poetryArgs;
    in {
      devShell = pkgs.mkShell rec {
        buildInputs = with pkgs; [
          cudatoolkit
          nvidia_x11
          poetry
          poetryEnv
          pre-commit
        ];
        shellHook = ''
          export PYTHONFAULTHANDLER=1
          export PYTHONBREAKPOINT=ipdb.set_trace
          set -o allexport
          source .env
          set +o allexport
          export CUDA_PATH=${cudatoolkit.lib}
          export LD_LIBRARY_PATH=${cudatoolkit.lib}/lib:${nvidia_x11}/lib
          export EXTRA_LDFLAGS="-l/lib -l${nvidia_x11}/lib"
          export EXTRA_CCFLAGS="-i/usr/include"
        '';
      };
      packages.default = poetryApp;
    };
  in
    utils.lib.eachDefaultSystem out;
}
