{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    utils = {
      url = "github:numtide/flake-utils";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    self,
    nixpkgs,
    utils,
  }: let
    out = system: let
      pkgs = import nixpkgs {
        inherit system;
        config.cudaSupport = true;
        config.allowUnfree = true;
        #overlays = [
        #(final: prev: {
        ## Reassigning python3 to python39 so that arrow-cpp
        ## will be built using it.
        #python3 = prev.python39.override {
        #packageOverrides = pyfinal: pyprev: {
        ## Dependency of arrow-cpp
        ## Work-around for PyOpenSSL marked as broken on aarch64-darwin
        ## See: https://github.com/NixOS/nixpkgs/pull/172397,
        ## https://github.com/pyca/pyopenssl/issues/87
        #pyopenssl =
        #pyprev.pyopenssl.overridePythonAttrs
        #(old: {meta.broken = false;});

        ## Twisted currently fails tests because of pyopenssl
        ## (see linked issues above)
        #twisted = pyprev.buildPythonPackage {
        #pname = "twisted";
        #version = "22.4.0";
        #format = "wheel";
        #src = final.fetchurl {
        #url = "https://files.pythonhosted.org/packages/db/99/38622ff95bb740bcc991f548eb46295bba62fcb6e907db1987c4d92edd09/Twisted-22.4.0-py3-none-any.whl";
        #sha256 = "sha256-+fepH5STJHep/DsWnVf1T5bG50oj142c5UA5p/SJKKI=";
        #};
        #propagatedBuildInputs = with pyfinal; [
        #automat
        #constantly
        #hyperlink
        #incremental
        #setuptools
        #typing-extensions
        #zope_interface
        #];
        #};
        #};
        #};
        #thrift = prev.thrift.overrideAttrs (old: {
        ## Concurrency test fails on Darwin
        ## TInterruptTest, TNonblockingSSLServerTest
        ## SecurityTest, and SecurityFromBufferTest
        ## fail on Linux.
        #doCheck = false;
        #});
        #})
        #];
      };
      inherit (pkgs) poetry2nix lib stdenv fetchurl;
      inherit (pkgs.cudaPackages) cudatoolkit;
      inherit (pkgs.linuxPackages) nvidia_x11;
      python = pkgs.python39;
      pythonEnv = poetry2nix.mkPoetryEnv {
        inherit python;
        projectDir = ./.;
        preferWheels = true;
        overrides =
          poetry2nix.overrides.withDefaults
          (pyfinal: pyprev: rec {
            # inherit (python.pkgs) apache-beam;
            # Use tensorflow-gpu on linux
            tensorflow-gpu =
              # Override the nixpkgs bin version instead of
              # poetry2nix version so that rpath is set correctly.
              pyprev.tensorflow-bin.overridePythonAttrs
              (old: {inherit (pyprev.tensorflow-gpu) src version;});
            #Use tensorflow-macos on macOS
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
    in {
      devShell = pkgs.mkShell {
        buildInputs = [pythonEnv nvidia_x11 cudatoolkit];
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
    };
  in
    with utils.lib; eachSystem defaultSystems out;
}
