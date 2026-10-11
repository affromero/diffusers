# Diffusers 0.32.2 security backport

This fork retains the 0.32.2 API and uses version `0.32.2+security.1`.
It backports the custom-code trust checks from upstream commit
`a37f6f8394ac2a7ee8360c3abea811efe54512b1` for
GHSA-7wx4-6vff-v64p and GHSA-98h9-4798-4q5v.

Local and Hub custom pipelines and components require
`trust_remote_code=True` before dependency imports or module execution.
The same decision applies to device-map loading, pipeline conversion, and
connected pipelines. Unknown component libraries also require explicit trust.
The canonical community pipeline mirror remains available by default.
`DIFFUSERS_DISABLE_REMOTE_CODE=1` rejects all custom code, including community
pipelines and explicitly trusted modules.

The loader classifies local files once and uses that classification for both
the trust decision and the loading branch. This prevents a newly created local
file from inheriting the community mirror exemption.

Regression tests exercise denied imports, trusted CPU inference, local
components, Hub relative imports, and filesystem replacement. CUDA generation
has not been tested because no GPU is available.
