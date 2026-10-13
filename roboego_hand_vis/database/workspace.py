"""Attach local workspace preparation/publication to a stage service."""


class WorkspaceService:
    def __init__(self, service, workspace):
        self.service, self.workspace = service, workspace
        self.work_root = workspace.local_root
        self._active_id = None

    def __getattr__(self, name):
        original = getattr(self.service, name)
        if name == 'claim_' + self.workspace.stage:
            def claim():
                row = original()
                if row is None:
                    return None
                self._active_id = row['id']
                try:
                    self.workspace.prepare(row['local_path'])
                except BaseException:
                    try:
                        self._write_failure(row['id'])
                    finally:
                        self.workspace.release()
                        self._active_id = None
                    raise
                return row
            return claim
        if name == 'finish_' + self.workspace.stage:
            def finish(data_id, **kwargs):
                if data_id != self._active_id:
                    raise RuntimeError('Workspace task ID does not match the claimed row')
                status = ('completed' if kwargs.get('succeeded') is True or kwargs.get('passed') is True
                          else 'block' if kwargs.get('passed', object()) is False else 'failed')
                try:
                    self.workspace.publish(status)
                    return original(data_id, **kwargs)
                except BaseException as exc:
                    self._write_failure(data_id)
                    if isinstance(exc, Exception):
                        raise RuntimeError(f'阶段结果同步或状态写回失败，已保留本地缓存：{exc}') from exc
                    raise
                finally:
                    self.workspace.release()
                    self._active_id = None
            return finish
        return original

    def _write_failure(self, data_id):
        from .stage_schema import STAGE_REQUIRED_COLUMNS
        # Gate writeback uses passed=None; ordinary stages use succeeded=False.
        stage = self.workspace.stage
        assert stage in STAGE_REQUIRED_COLUMNS
        kw = {'passed': None} if stage in {
            'raw_headband_integrity', 'video_simple_quality', 'stereo_depth_gate'
        } else {'succeeded': False}
        if not getattr(self.service, 'finish_' + stage)(data_id, **kw):
            raise RuntimeError('无法写回失败状态，请检查数据库中的 processing 任务')
