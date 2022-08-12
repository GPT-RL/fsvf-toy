# `fetchPypi` function for fetching artifacts from PyPI.
{
  fetchurl,
  makeOverridable,
}: let
  computeUrl = {format ? "setuptools", ...} @ attrs: let
    computeWheelUrl = {
      pname,
      version,
      dist ? "py2.py3",
      python ? "py2.py3",
      abi ? "none",
      platform ? "any",
    }:
    # Fetch a wheel. By default we fetch an universal wheel.
    # See https://www.python.org/dev/peps/pep-0427/#file-name-convention for details regarding the optional arguments.
    "https://files.pythonhosted.org/packages/${dist}/${builtins.substring 0 1 pname}/${pname}/${pname}-${version}-${python}-${abi}-${platform}.whl";

    computeSourceUrl = {
      pname,
      version,
      extension ? "tar.gz",
    }:
    # Fetch a source tarball.
    "mirror://pypi/${builtins.substring 0 1 pname}/${pname}/${pname}-${version}.${extension}";

    compute =
      if format == "wheel"
      then computeWheelUrl
      else if format == "setuptools"
      then computeSourceUrl
      else throw "Unsupported format ${format}";
  in
    compute (builtins.removeAttrs attrs ["format"]);
in
  makeOverridable ({
      format ? "setuptools",
      sha256 ? "",
      hash ? "",
      ...
    } @ attrs: let
      _url = computeUrl (builtins.removeAttrs attrs ["sha256" "hash"]);
      url = "https://files.pythonhosted.org/packages/c1/28/223754a43279d6c5d66e89aa8cff7b899a895867f14966593286ff8896bd/clu-0.0.7-py3-none-any.whl";
    in
      fetchurl {
        inherit url sha256 hash;
      })
