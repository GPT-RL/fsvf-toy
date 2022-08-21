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
          inherit (pyprev) absl-py flatbuffers numpy scipy six;
        };
        ml-collections = pyprev.buildPythonPackage rec {
          pname = "ml_collections";
          version = "0.1.1";
          src = pyprev.fetchPypi {
            inherit pname version;
            sha256 = "sha256-P+/McuxDOqHl0yMHo+R0u7Z/QFvoFOpSohZr/J2+aMw=";
          };
          buildInputs = with pyfinal; [absl-py contextlib2 pyyaml six];
          prePatch = ''
            export HOME=$TMPDIR;
          '';
        };
        ray = pyprev.ray.overridePythonAttrs (old: {
          propagatedBuildInputs =
            (old.propagatedBuildInputs or [])
            ++ [pyfinal.pandas];
        });
        run-logger = pyprev.run-logger.overridePythonAttrs (old: {
          buildInputs = old.buildInputs or [] ++ [pyprev.poetry];
        });
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
      buildInputs = with pkgs; [
        alejandra
        coreutils
        nodePackages.prettier
        poetry
        poetryEnv
      ];
    in rec {
      devShell = pkgs.mkShell rec {
        inherit buildInputs;
        PYTHONFAULTHANDLER = 1;
        PYTHONBREAKPOINT = "ipdb.set_trace";
        LD_LIBRARY_PATH = "${nvidia_x11}/lib";
        shellHook = ''
          set -o allexport
          source .env
          set +o allexport
        '';
      };
      packages.default = pkgs.dockerTools.buildImage {
        name = "ppo";
        tag = "latest";
        copyToRoot =
          pkgs.buildEnv
          {
            name = "image-root";
            pathsToLink = ["/bin" "ppo"];
            paths = buildInputs ++ [./ppo];
          };
        config = {
          Env = with pkgs; [
            "PYTHONFAULTHANDLER=1"
            "PYTHONBREAKPOINT=ipdb.set_trace"
            "LD_LIBRARY_PATH=/usr/lib64/"
            "PATH=/bin:$PATH"
          ];
          Cmd = ["${pkgs.bash}/bin/bash"];
        };
      };
    };
  in
    utils.lib.eachDefaultSystem out;
}
