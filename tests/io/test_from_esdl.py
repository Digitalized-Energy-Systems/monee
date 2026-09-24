from types import SimpleNamespace

from monee.io.from_esdl import _walk_assets


def test_walk_assets_yields_each_asset_once():
    area = SimpleNamespace(asset=[SimpleNamespace(name="a"), SimpleNamespace(name="b")])

    names = [asset.name for asset in _walk_assets(area)]

    assert names == ["a", "b"]


def test_walk_assets_recurses_into_container_assets():
    inner = SimpleNamespace(name="inner")
    building = SimpleNamespace(name="building", asset=[inner])
    area = SimpleNamespace(asset=[SimpleNamespace(name="top"), building])

    names = [asset.name for asset in _walk_assets(area)]

    assert names == ["top", "building", "inner"]


def test_walk_assets_recurses_into_sub_areas():
    sub_area = SimpleNamespace(asset=[SimpleNamespace(name="sub")])
    area = SimpleNamespace(
        asset=[SimpleNamespace(name="top")],
        area=[sub_area],
    )

    names = [asset.name for asset in _walk_assets(area)]

    assert names == ["top", "sub"]


def test_walk_assets_handles_missing_attributes():
    assert list(_walk_assets(SimpleNamespace())) == []
