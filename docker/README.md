# Docker Environments

Run all documented commands from the roboego-hand-vis repository root.

| Image | Build and run instructions | Verified scope |
| --- | --- | --- |
| `roboego-hand-env:dev` | [Hand README](hand-env/README.md) | Offline MINT + canonical export + EgoHandKit, documented parameters, 100 frames |
| `roboego-vio-env:dev` | [VIO README](vio-env/README.md) | Offline replay and real-data outputs verified; cashier trajectory diverges, quality acceptance pending |

These are environment images, not copies of the algorithm repositories. Source
is mounted read-only from the host, while results are written to separate host
directories. Each launcher creates a temporary container and removes it after
the command finishes. Outputs, model volumes and build directories persist.
Python source changes take effect on the next task; native changes require
recompilation. Dependency or bundled model changes require rebuilding the image.

The hand image includes separately licensed model assets. This Git branch
contains build scripts, documentation and tests only: no weights, input samples,
rendered results, Docker image archives or external algorithm repositories.
Building on another machine requires preparing those repositories/assets first.
Do not publish model-containing images before checking redistribution rights.

## Moving Built Images Offline

After building, privately transfer an image archive to the target machine:

```bash
docker save -o /path/outside/repository/hand-env.tar roboego-hand-env:dev
docker save -o /path/outside/repository/vio-env.tar roboego-vio-env:dev
# On the destination machine:
docker load -i /path/to/hand-env.tar
docker load -i /path/to/vio-env.tar
```

The destination still needs Docker, the corresponding source working trees and
input data. The hand image additionally needs a compatible NVIDIA driver and
NVIDIA Container Toolkit. Image export does not capture mounted code or outputs.
Record source revisions (including local changes) alongside the image ID.
