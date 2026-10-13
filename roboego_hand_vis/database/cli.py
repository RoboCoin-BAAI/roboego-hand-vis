"""Shared table selection and stage schema preflight for database entry points."""


def add_table_argument(parser):
    parser.add_argument('--local-root', help='可选本地工作根目录；省略时直接在共享目录处理。')
    parser.add_argument('--table', type=str.lower, choices=('l1data', 'l2data', 'l3data'),
                        default='l1data', help='领取及更新的数据表，默认 l1data。')


def stage_service_or_error(parser, url, table_name, stage, *, args=None):
    from .data_service import DataService
    workspace = None
    if args is not None:
        from pathlib import Path
        if args.local_root:
            from roboego_hand_vis.common.session_workspace import SessionWorkspace
            try:
                workspace = SessionWorkspace(args.shared_root, args.local_root, stage,
                                             copy_missing=stage == 'raw_headband_integrity')
            except (ValueError, OSError) as exc:
                parser.error(str(exc))
        work_root = workspace.local_root if workspace else Path(args.shared_root).expanduser().resolve()
    service = DataService.from_url(url, table_name=table_name)
    try:
        service.validate_stage(stage)
    except Exception as exc:
        service.database.close()
        # Avoid putting connection URLs or credentials into CLI errors.
        if isinstance(exc, ValueError):
            parser.error(str(exc))
        parser.error(f'无法检查表 {table_name} 的结构，请检查数据库连接和表是否存在。')
    if args is not None:
        args.shared_root = str(work_root)
        if workspace:
            from .workspace import WorkspaceService
            service = WorkspaceService(service, workspace)
        else:
            service.work_root = work_root
    return service
