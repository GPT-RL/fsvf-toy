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
      overrides = pyfinal: pyprev: rec {
        astunparse = pyprev.astunparse.overridePythonAttrs (old: {
          buildInputs = (old.buildInputs or []) ++ [pyfinal.wheel];
        });
        clu = pyprev.buildPythonPackage rec {
          pname = "clu";
          version = "0.0.7";
          src = pyprev.fetchPypi {
            inherit pname version;
            sha256 = "sha256-RJqa8XnDpcRPwYlH+4RKAOos0x4+3hMWf/bv6JNn2ys=";
          };
          buildInputs = with pyfinal; [
            absl-py
            etils
            flax
            jax
            jaxlib
            ml-collections
            numpy
            packaging
            tensorflow
            tensorflow-datasets
          ];
        };
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
        ml-collections =
          pyprev.buildPythonPackage
          rec {
            pname = "ml_collections";
            version = "0.1.1";
            src = pyprev.fetchPypi {
              inherit pname version;
              sha256 = "sha256-P+/McuxDOqHl0yMHo+R0u7Z/QFvoFOpSohZr/J2+aMw=";
            };
            buildInputs = with pyfinal; [
              absl-py
              contextlib2
              pyyaml
              six
            ];
            prePatch = ''
              export HOME=$TMPDIR;
            '';
          };
        tensorflow-gpu =
          # Override the nixpkgs bin version instead of
          # poetry2nix version so that rpath is set correctly.
          pyprev.tensorflow-bin.overridePythonAttrs
          (old: {inherit (pyprev.tensorflow-gpu) src version;});
      };
      poetryArgs = {
        inherit python;
        projectDir = ./.;
        preferWheels = true;
        overrides = poetry2nix.overrides.withDefaults overrides;
      };
      poetryEnv = mkPoetryEnv poetryArgs;
    in {
      devShell = pkgs.mkShell rec {
        buildInputs = with pkgs; [
          cudatoolkit
          nvidia_x11
          poetry
          poetryEnv
          pre-commit
          nodePackages.prettier
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
      packages.default = mkPoetryApplication poetryArgs;
    };
  in
    utils.lib.eachDefaultSystem out;
}
