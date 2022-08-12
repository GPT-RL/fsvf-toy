{
  lib,
  buildPythonPackage,
  fetchPypi,
  pytypeclass,
}:
buildPythonPackage rec {
  pname = "dollar-lambda";
  version = "1.1.4";
  src = fetchPypi {
    inherit pname version;
    sha256 = "sha256-dq2l4h+uRr8jkr6fZu33pxvFzHzh0lP4p7+gZmEwqDg=";
  };
  propagatedBuildInputs = [pytypeclass];

  meta = with lib; {
    homepage = "https://dollar-lambda.readthedocs.io/";
    description = "An argument parser for Python built from functional first principles";
    longDescription = ''
      $λ provides an alternative to argparse based on parser combinators and functional first principles. Arguably, $λ is way more expressive than any reasonable person would ever need... but even if it's not the parser that we need, it's the parser we deserve.
    '';
    license = licenses.mit;
    maintainers = with maintainers; [ethanabrooks];
  };
}
