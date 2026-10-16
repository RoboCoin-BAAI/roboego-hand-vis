# VIO runtime unification

Goal: use the project VIO CLI as the sole owner of preparation and delivery, for both database and manual runs.

Approved design: retain the existing SESSION OUTPUT run interface and database locking. Add a repository-owned deployment script that calls the project CLI directly inside rootfs, bypassing runtime-output.py. Preserve timing environment defaults, forward CLI arguments, log the resolved configuration, and copy output and failure logs back. Do not alter OpenVINS or the user's YAML changes.

- [x] Add shell integration tests with a fake chroot executable (no privileged execution), covering parameter forwarding, output publication, failure exit/log preservation, and refusal to overwrite existing paths.
- [x] Implement deploy/vio-run.sh with guarded input staging, rootfs locking, direct Python entry and output copying.
- [x] Log resolved configuration in cli.py; test CLI override precedence.
- [x] Document deployment, manual runs, configuration ownership and removal of the old wrapper from the call chain.
- [x] Run VIO tests, shell syntax check and review diff. Actual server/rootfs execution remains a deployment validation step.

Validation: 47 VIO tests passed; bash -n and git diff --check passed. Independent review found a pre-lock output check race; moved check under rootfs lock and added regression coverage. Real server/chroot execution has not been performed.
