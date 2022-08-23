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
      inherit (pkgs) bash buildEnv cudaPackages dockerTools linuxPackages mkShell poetry2nix python39 stdenv;
      inherit (poetry2nix) mkPoetryApplication mkPoetryEnv;
      inherit (cudaPackages) cudatoolkit;
      inherit (linuxPackages) nvidia_x11;
      python = python39;
      overrides = pyfinal: pyprev: let
        inherit (pyprev) buildPythonPackage fetchPypi;
      in rec {
        ale-py = pyprev.ale-py.overridePythonAttrs (let
          roms = fetchTarball {
            url = "https://roms8.s3.us-east-2.amazonaws.com/Roms.tar.gz";
            sha256 = "sha256:0g9xffdm7zndf84m14f1j1x1v3ybm7ls7498071xlhah9k80bskq";
          };
        in {
          postInstall = ''
            export LD_LIBRARY_PATH=${stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
            $out/bin/ale-import-roms ${roms}
          '';
        });
        astunparse = pyprev.astunparse.overridePythonAttrs (old: {
          buildInputs = (old.buildInputs or []) ++ [pyfinal.wheel];
        });
        clu = buildPythonPackage rec {
          pname = "clu";
          version = "0.0.7";
          src = fetchPypi {
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
        etils = pyprev.etils.overridePythonAttrs (old: {
          propagatedBuildInputs =
            builtins.filter (i: i.pname != "etils") old.propagatedBuildInputs;
        });
        # Use cuda-enabled jaxlib as required
        jaxlib = pyprev.jaxlibWithCuda.override {
          inherit (pyprev) absl-py flatbuffers numpy scipy six;
        };
        ml-collections = buildPythonPackage rec {
          pname = "ml_collections";
          version = "0.1.1";
          src = fetchPypi {
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
          {inherit (pyprev.tensorflow-gpu) src version;};
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
      devShell = mkShell rec {
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
      packages.default = dockerTools.buildImage {
        name = "ppo";
        tag = "latest";
        copyToRoot =
          buildEnv
          {
            name = "image-root";
            pathsToLink = ["/bin" "/ppo"];
            paths = buildInputs ++ [pkgs.git ./.];
          };
        config = {
          Env = with pkgs; [
            "PYTHONFAULTHANDLER=1"
            "PYTHONBREAKPOINT=ipdb.set_trace"
            "LD_LIBRARY_PATH=/usr/lib64/"
            "PATH=/bin:$PATH"
          ];
          Cmd = ["${poetryEnv.python}/bin/python"];
        };
      };
    };
  in
    utils.lib.eachDefaultSystem out;
}
