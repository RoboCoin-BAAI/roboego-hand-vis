import pytest


def test_completion_writeback_failure_stops_claiming(tmp_path):
    from roboego_hand_vis.download_data.database_adapter import download_pending_l1
    from roboego_hand_vis.download_data.downloader import DownloadResult

    class Service:
        def __init__(self):
            self.claims = 0

        def claim(self, table):
            self.claims += 1
            return dict(id=1, ks3_path="ks3://bucket/session")

        def complete(self, *args):
            return False

    class Downloader:
        def download(self, source):
            return DownloadResult(tmp_path / "session", "session")

    service = Service()
    with pytest.raises(RuntimeError, match="completion was rejected"):
        download_pending_l1(service, Downloader())
    assert service.claims == 1
