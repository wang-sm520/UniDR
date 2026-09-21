"""Portable retrospective comparison of fixed checkpoint videos and native scores."""

from __future__ import annotations

import csv
import html
import importlib
import json
import math
from hashlib import sha256
from pathlib import Path
from typing import Any

SOURCES = ("motrix", "isaacsim", "isaacgym", "genesis", "joint")
COLORS = dict(zip(SOURCES, ("C3", "C0", "C1", "C2", "C4")))
LABELS = dict(
    zip(SOURCES, ("Motrix 单源", "Isaac Sim 单源", "Isaac Gym 单源", "Genesis 单源", "四源联合"))
)
SCALES = dict(
    joint_rmse_rad=1.0,
    root_position_error_m=1.0,
    root_orientation_error_rad=math.pi,
    body_position_rmse_m=0.5,
    ee_height_rmse_m=0.5,
)
OUTPUTS = ("REPORT.md", "index.html", "checkpoint_summary.csv", "report_manifest.json", "figures")
ARTIFACTS = "metrics.json native/verification.json native/physics_snapshots.npz native/telemetry.jsonl native/mujoco_config.json video/verification.json".split()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(path.read_text())
    return result


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local(root: Path, relative: str) -> Path:
    path = Path(relative)
    _require(not path.is_absolute() and ".." not in path.parts, "Unsafe artifact path")
    target = (root / path).resolve(strict=True)
    _require(root in target.parents, "Artifact escapes report directory")
    return target


def _finite(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)
    elif isinstance(value, float):
        _require(math.isfinite(value), "Non-finite report metric")


def expected_checkpoints() -> list[tuple[str, int]]:
    return [
        (source, i)
        for source in SOURCES
        for i in (
            *range(0, 5000 if source == "joint" else 20000, 500),
            4999 if source == "joint" else 19999,
        )
    ]


def _validate(root: Path) -> tuple[dict, dict, list[dict], dict]:
    expected = expected_checkpoints()
    paths = [f"checkpoints/{source}/model_{index:05d}/result.json" for source, index in expected]
    found = {str(p.relative_to(root)) for p in root.glob("checkpoints/*/model_*/result.json")}
    _require(
        found == set(paths),
        f"Incomplete checkpoint set: expected 175, found {len(found)}; missing {sorted(set(paths) - found)[:3]}",
    )
    training = _read(_local(root, "training/intervals.json"))
    config = _read(_local(root, "audit/config-comparison.json"))
    _finite(training)
    _finite(config)
    _require(config["audit_passed"] is True, "Configuration audit failed")
    clocks = {(r["policy"], r["iteration"]): r for r in training["checkpoint_clocks"]}
    _require(
        len(training["checkpoint_clocks"]) == len(clocks) == 175 and set(clocks) == set(expected),
        "Training checkpoint clocks incomplete or duplicated",
    )
    _require(
        training["interval_updates"] == 500
        and len(training["intervals"]) == 170
        and len(training["source_intervals"]) == 200,
        "Training interval coverage mismatch",
    )
    for source in SOURCES:
        stop = 5000 if source == "joint" else 20000
        rows = [r for r in training["intervals"] if r["policy"] == source]
        _require(
            [(r["first_completed_update"], r["last_completed_update"]) for r in rows]
            == [(i + 1, i + 500) for i in range(0, stop, 500)],
            "Non-contiguous training intervals",
        )
    artifacts: dict[str, str] = {}
    results: list[dict] = []
    for (source, index), relative in zip(expected, paths, strict=True):
        result = _read(_local(root, relative))
        _finite(result)
        _require(
            result["protocol"].startswith("retrospective-500-")
            and (not results or result["protocol"] == results[0]["protocol"])
            and result["source"] == source
            and type(result["checkpoint_index"]) is int
            and result["checkpoint_index"] == index,
            "Checkpoint identity mismatch",
        )
        _require(
            result["completed_updates"] == index + 1
            and result["planned_updates"] == (5000 if source == "joint" else 20000),
            "Checkpoint update mismatch",
        )
        _require(
            result["total_transitions"] == (index + 1) * (98304 if source == "joint" else 24576)
            and result["optimizer_steps"] == (index + 1) * 20,
            "Checkpoint budget mismatch",
        )
        clock = clocks[source, index]
        _require(
            clock["cumulative_transitions"] == result["total_transitions"]
            and clock["completed_updates"] == result["completed_updates"]
            and clock["cumulative_optimizer_steps"] == result["optimizer_steps"],
            "Checkpoint/log budget mismatch",
        )
        prefix = f"checkpoints/{source}/model_{index:05d}/"
        _require(
            result["video"] == f"media/{source}_{index:05d}.mp4"
            and result["poster"] == f"media/{source}_{index:05d}.jpg",
            "Media identity mismatch",
        )
        required = {result["video"], result["poster"], *(prefix + p for p in ARTIFACTS)}
        _require(required <= result["artifact_sha256"].keys(), "Required artifact digest missing")
        for artifact, digest in result["artifact_sha256"].items():
            _require(
                _hash(_local(root, artifact)) == digest, f"Artifact digest mismatch: {artifact}"
            )
            _require(
                artifact not in artifacts or artifacts[artifact] == digest,
                "Conflicting artifact digest",
            )
            artifacts[artifact] = digest
        artifacts[relative] = _hash(root / relative)
        _require(
            result["metrics"] == _read(root / prefix / "metrics.json"),
            "Inline and persisted metrics differ",
        )
        native = _read(root / prefix / "native/verification.json")
        _require(
            native["checkpoint_sha256"] == result["checkpoint_sha256"]
            and native["backend"] == "mujoco"
            and native["strict_preflight"] is True
            and native["actor_normalizer_unchanged"] is True
            and native["seed"] == 1,
            "Native evaluation provenance mismatch",
        )
        _require(
            result["video_format"]
            == dict(frames=1000, width=1280, height=720, fps=50, seconds=20.0),
            "Unexpected video format",
        )
        metrics = result["metrics"]
        score = metrics["fixed_first_opportunity"]
        _require(
            metrics["protocol"] == "g1_native_tracking_v1"
            and metrics["frames"] == 1000
            and metrics["control_dt_seconds"] == 0.02,
            "Scoring protocol mismatch",
        )
        _require(
            score["horizon_steps"] == 224
            and score["horizon_seconds"] == 4.48
            and score["error_scales"] == SCALES,
            "Fixed horizon or normalization scales differ",
        )
        _require(
            0 <= score["survived_step_fraction"] <= 1
            and score["observed_steps_including_failure"] + score["padded_steps_after_failure"]
            == 224,
            "Invalid scoring coverage",
        )
        _require(
            set(score["normalized_error_with_failure_padding"]) == set(SCALES)
            and all(0 <= v <= 1 for v in score["normalized_error_with_failure_padding"].values()),
            "Invalid normalized errors",
        )
        _require(
            result["video_mode"]
            == ("failure-tail-3.5s" if metrics["true_terminations"] else "native"),
            "Native score and diagnostic video mode disagree",
        )
        results.append({**result, "cumulative_loop_seconds": clock["cumulative_loop_seconds"]})
    for relative in (
        "training/intervals.json",
        "audit/config-comparison.json",
        *(
            f"training/{name}.csv"
            for name in ("intervals", "source_intervals", "scalar_intervals", "checkpoint_clocks")
        ),
        *(
            f"training/{name}.{suffix}"
            for name in ("episodes", "tracking", "timing_ppo")
            for suffix in ("png", "pdf")
        ),
    ):
        artifacts[relative] = _hash(_local(root, relative))
    return training, config, results, artifacts


def _flatten(value: dict, prefix: str = "") -> dict:
    result = {}
    for key, item in value.items():
        if key in ("input_sha256", "semantics"):
            continue
        name = prefix + key
        if isinstance(item, dict):
            result.update(_flatten(item, name + "."))
        else:
            result[name] = json.dumps(item, ensure_ascii=False) if isinstance(item, list) else item
    return result


def _plots(root: Path, results: list[dict]) -> None:
    importlib.import_module("matplotlib").use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")
    specs = [
        ("samples", "total_transitions", 1e6, "Global transitions (millions)"),
        ("updates", "completed_updates", 1, "Completed PPO updates"),
        ("time", "cumulative_loop_seconds", 3600, "Recorded training loop hours"),
    ]
    metrics = [
        ("survived_step_fraction", "Survived step fraction"),
        ("return_with_zero_failure_padding", "First-opportunity return"),
        *(
            (name, name)
            for name in (
                "joint_rmse_rad",
                "body_position_rmse_m",
                "root_orientation_error_rad",
                "ee_height_rmse_m",
            )
        ),
    ]
    with plt.style.context("seaborn-v0_8-whitegrid"):
        for suffix, key, divisor, xlabel in specs:
            fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
            for ax, (metric, label) in zip(axes.flat, metrics, strict=True):
                for source in SOURCES:
                    rows = [r for r in results if r["source"] == source]
                    scores = [r["metrics"]["fixed_first_opportunity"] for r in rows]
                    values = [
                        s[metric]
                        if metric in s
                        else s["normalized_error_with_failure_padding"][metric]
                        for s in scores
                    ]
                    ax.plot(
                        [r[key] / divisor for r in rows],
                        values,
                        marker=".",
                        markersize=3,
                        linewidth=1,
                        label=source,
                        color=COLORS[source],
                    )
                note = "\n(normalized, dimensionless, failure padded)" if metric in SCALES else ""
                ylabel = label.removesuffix("_rad").removesuffix("_m").replace("_", " ") + note
                ax.set(xlabel=xlabel, ylabel=ylabel)
                ax.legend(fontsize=7)
            fig.suptitle(
                "Fixed 224-step native MuJoCo opportunity | retrospective, one seed, no model selection"
            )
            for extension in ("png", "pdf"):
                fig.savefig(root / "figures" / f"holdout_{suffix}.{extension}", dpi=150)
            plt.close(fig)


def _number(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _table(headers: list[str], rows: list[list]) -> str:
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{html.escape(h)}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{html.escape(_number(v))}</td>" for v in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table>\n"
    )


def _link(path: str, label: str) -> str:
    return f'<a href="{html.escape(path, quote=True)}">{html.escape(label)}</a>'


def _document(training: dict, config: dict, results: list[dict]) -> str:
    sections = []

    def text(value: str) -> None:
        sections.append("<p>" + html.escape(value) + "</p>")

    def heading(value: str) -> None:
        sections.append("<h2>" + html.escape(value) + "</h2>")

    text(
        "已完成实验的事后复盘：四种单一仿真训练与四源联合训练，共 175 个已保存 checkpoint，全部提供 MuJoCo 视频和同协议量化记录。结果不回流训练，不用于调参、挑选 checkpoint 或改变固定最终模型。"
    )
    text(
        "解压完整目录后，用浏览器打开 index.html；所有视频与图片均为相对链接，无需网络。若 Markdown 阅读器屏蔽 video 标签，点击封面或视频链接。每条视频 20 秒、1280×720、50 FPS，经典 MuJoCo 背景、正面策略机器人及青色参考、无文字叠加；没有真实硬件实验。"
    )
    heading("比较条件与解释边界")
    text(
        "任务为 G1FlipTracking，G1 29 DoF，actor/critic 观测 160/286，29 动作；动作文件 flip_360_001__A304.npz。任务奖励、动作缩放、参考相位、原生终止和网络沿用训练保存配置；物理 dt=0.005 s，控制 dt=0.02 s，24 步/窗口，512→256→128 ELU，经验归一化与 adaptive KL PPO，5 epochs×4 minibatches，每轮20次优化。"
    )
    text(
        "没有额外物理参数或观测 DR：reset 位置、姿态、速度、关节扰动为零，参考从第0帧开始；PPO随机动作和开训时超时计数错开不属于环境DR。四源联合使用固定25%配额与四个固定动力学域。全部关闭自碰撞，保留地面接触；terminated和timeout同时触发时真正终止优先，不bootstrap，纯timeout仍bootstrap。"
    )
    text(
        "各单源1024环境×20000更新，联合每源1024、总4096环境×5000更新。每个最终策略都是491,520,000 transitions；单源400,000次optimizer steps，联合100,000次。联合每轮batch=98,304，是单源24,576的4倍；不能把同轮数视为同采样量，不能将最终等样本比较说成等优化预算。"
    )
    text(
        "checkpoint文件下标从0开始：model_500是完成501次更新，model_19999/4999是完成20000/5000次更新。图使用真实样本坐标，不把单源model_2000与联合model_500误称严格等量，也不插值制造配对；只有最终端点样本完全相等。训练耗时是日志collection+learning，不含初始化和部分保存/日志开销；原生GAE计入learning，联合GAE计入collection；联合分源耗时重叠，不能相加代替总墙钟时间。"
    )
    sections.append(
        _link("audit/config-comparison.json", "完整配置、奖励、DR、后端差异和资产哈希审计")
    )
    sections.append(
        "<details><summary>展开实际配置审计摘要（完整数据见链接）</summary><pre>"
        + html.escape(
            json.dumps(
                {k: v for k, v in config.items() if k != "shared"}
                | {
                    "shared": {
                        k: v for k, v in config.get("shared", {}).items() if k != "asset_hashes"
                    }
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        + "</pre></details>"
    )
    heading("MuJoCo 主评分与视频协议")
    text(
        "每个模型从相同初始条件和参考相位、seed=1，使用确定性actor与冻结normalizer，在原生终止规则下独立rollout。主评分只使用首个224控制步（4.48s），截止参考第224帧，避开第225控制步的参考循环物理重置；不能把reset当成功落地。"
    )
    text(
        "真正失败的terminal observation、error和reward计入；此后剩余时域的归一误差补1、reward补0，不从下一episode补样。每项误差先除固定尺度并clip到[0,1]，再按224步平均，结果无量纲。survived_step_fraction为失败前存活步数/224；整段不失败为1。表中首失败列限于224步；首轮不失败也可能在20秒内后续循环失败，后者记录在CSV的first_failure_seconds。存活比例不等于空翻成功率，也不等于起跳/翻转/落地认证。"
    )
    sections.append(_table(["归一误差", "固定尺度"], [[k, v] for k, v in SCALES.items()]))
    text(
        "视频与主评分严格区分：标为原生的录像沿用原生reset；出现失败的模型提供延后reset 3.5秒的诊断录像，继续策略控制，以看到触地、翻滚和倒地。首次原生失败前保持正常参考循环；仅首次失败之后允许在参考末帧保持。固定20秒结尾可能截断最后一次失败的尾段，具体见video/verification.json的incomplete_final_tail，不能保证每次末尾摔倒都录满3.5秒。诊断尾段不进入主评分，不能据该录像的reset时刻反推评分。20秒全轨迹误差、动作RMS、动作变化、终止原因、episode与翻转角度也导出，但受存活时长/相位占用影响，只作描述指标。"
    )
    text(
        "每个checkpoint仅一个固定seed；重复参考片段不是独立试验，不能据此报告统计成功率。landing_candidate是运动学候选，未提供接触力成功认证；矢状面累计翻转角不等于完整3D旋转指标。四GPU实机、长期多seed效果、GPU峰值显存、能耗、训练期间全轨迹误差未被这批日志测得，不补造。"
    )
    heading("固定最终模型对比")
    finals = [r for r in results if r["checkpoint_index"] == r["planned_updates"] - 1]
    sections.append(
        _table(
            [
                "策略",
                "完成更新",
                "样本",
                "优化次数",
                "训练loop小时",
                "224步存活比例",
                "归一body误差",
                "224步回报",
                "20秒真正失败次数",
                "landing候选数",
            ],
            [
                [
                    LABELS[r["source"]],
                    r["completed_updates"],
                    r["total_transitions"],
                    r["optimizer_steps"],
                    r["cumulative_loop_seconds"] / 3600,
                    r["metrics"]["fixed_first_opportunity"]["survived_step_fraction"],
                    r["metrics"]["fixed_first_opportunity"][
                        "normalized_error_with_failure_padding"
                    ]["body_position_rmse_m"],
                    r["metrics"]["fixed_first_opportunity"]["return_with_zero_failure_padding"],
                    r["metrics"]["true_terminations"],
                    sum(len(a["landing_candidate_steps"]) for a in r["metrics"]["attempts"]),
                ]
                for r in finals
            ],
        )
    )
    heading("曲线与机器可读数据")
    for name in ("samples", "updates", "time"):
        sections.append(
            f'<p>{_link(f"figures/holdout_{name}.pdf", "PDF")}</p><img src="figures/holdout_{name}.png" alt="MuJoCo {name} comparison" width="1200">'
        )
    for name in ("episodes", "tracking", "timing_ppo"):
        sections.append(
            f'<p>{_link(f"training/{name}.pdf", "PDF")}</p><img src="training/{name}.png" alt="Training {name}" width="1200">'
        )
    for path, label in (
        ("checkpoint_summary.csv", "全部175个MuJoCo量化指标/episode/attempt与checkpoint哈希"),
        ("training/intervals.csv", "170个500次更新区间：时间、PPO、预算"),
        ("training/source_intervals.csv", "200个源区间：回报、12奖励项、10跟踪误差、联合源计时"),
        (
            "training/scalar_intervals.csv",
            "原始TensorBoard所有可用指标：均值/标准差/极值/P95/有效数/缺测数",
        ),
        ("training/checkpoint_clocks.csv", "全部checkpoint准确训练时间与预算"),
        ("training/intervals.json", "训练完整审计与指标语义"),
        ("audit/EXECUTION.md", "代码、实际 SHA、精确命令、验证结果与复现说明"),
        ("audit/training-findings.md", "训练曲线与耗时的数值解读"),
        ("audit/series-evidence-final.json", "175个checkpoint与视频的独立完整性、解码校验"),
        ("report_manifest.json", "本报告所有输入/输出哈希"),
    ):
        sections.append("<p>" + _link(path, label) + "</p>")
    text(
        "训练tracking/error日志仅在reset时测量、聚合，不是全轨迹时均误差。reward/*是加权pre-dt rate；Episode_Reward/*是episode sum除固定最大episode秒数；Train/mean_reward是已结束episode滚动回报。Episode_Termination/*是reset-step原因计数均值，不是失败率，也不能累加为次数。CSV保留缺测，不补零；跨更新标准差描述过程波动，不是多seed不确定性。"
    )
    heading("每500次真实完成更新的训练区间")
    interval_columns = "collection_seconds learning_seconds loop_seconds event_boundary_elapsed_seconds cumulative_loop_seconds transitions_per_loop_second episode_return_mean episode_length_mean Loss/value_mean Loss/entropy_mean Loss/learning_rate_mean".split()
    for source in SOURCES:
        rows = [r for r in training["intervals"] if r["policy"] == source]
        sections.append(f"<details><summary>{LABELS[source]}：{len(rows)} 个完整区间</summary>")
        sections.append(
            _table(
                [
                    "完成更新区间",
                    "采集秒",
                    "学习秒",
                    "总秒",
                    "事件边界墙钟秒（首区间缺测）",
                    "累计秒",
                    "samples/s",
                    "rolling return",
                    "episode length",
                    "value loss",
                    "entropy",
                    "学习率",
                ],
                [
                    [
                        f"{r['first_completed_update']}–{r['last_completed_update']}",
                        *(r[k] for k in interval_columns),
                    ]
                    for r in rows
                ],
            )
        )
        sections.append("</details>")
    heading("全部checkpoint索引与175条视频")
    for source in SOURCES:
        rows = [r for r in results if r["source"] == source]
        sections.append(f'<h3 id="{source}">{LABELS[source]}</h3>')
        table, previous = [], 0.0
        for r in rows:
            score = r["metrics"]["fixed_first_opportunity"]
            table.append(
                [
                    r["checkpoint_index"],
                    r["completed_updates"],
                    r["total_transitions"],
                    r["cumulative_loop_seconds"],
                    r["cumulative_loop_seconds"] - previous,
                    score["survived_step_fraction"],
                    score["normalized_error_with_failure_padding"]["body_position_rmse_m"],
                    score["return_with_zero_failure_padding"],
                    score["time_to_first_failure_seconds"],
                    score["first_failure_reference_frame"],
                    ", ".join(score["first_failure_terms"]) or "无",
                ]
            )
            previous = r["cumulative_loop_seconds"]
        sections.append(
            _table(
                [
                    "checkpoint下标",
                    "完成更新",
                    "全局samples",
                    "累计loop秒",
                    "距前checkpoint秒",
                    "224步存活",
                    "归一body误差",
                    "224步return",
                    "224步内首失败秒",
                    "224步内首失败参考帧",
                    "224步内首失败原因",
                ],
                table,
            )
        )
        for r in rows:
            index = r["checkpoint_index"]
            mode = "原生播放" if r["video_mode"] == "native" else "失败后延迟reset 3.5秒的诊断播放"
            sections.append(
                f"<details><summary>{LABELS[source]} · model_{index}.pt · 完成 {index + 1} 更新 · {mode}</summary>"
            )
            video, poster = (
                html.escape(r["video"], quote=True),
                html.escape(r["poster"], quote=True),
            )
            sections.append(
                f'<video controls preload="none" playsinline width="960" poster="{poster}"><source src="{video}" type="video/mp4">{_link(r["video"], "打开视频")}</video>'
            )
            sections.append(
                f'<p><a href="{video}"><img src="{poster}" alt="点击封面打开视频" width="320" loading="lazy"></a></p>'
            )
            sections.append(
                "<p>"
                + _link(r["video"], "直接打开 / 下载 MP4")
                + " · "
                + _link(
                    f"checkpoints/{source}/model_{index:05d}/result.json", "本模型完整指标与哈希"
                )
                + "</p></details>"
            )
    return "\n\n".join(sections)


def report(output_root: str | Path) -> dict:
    """Refuse incomplete evidence; build local HTML/Markdown only after all checks."""
    root = Path(output_root).resolve(strict=True)
    _require(not any((root / name).exists() for name in OUTPUTS), "Report output already exists")
    training, config, results, artifacts = _validate(root)
    body = _document(training, config, results)
    rows = []
    metadata_columns = "source checkpoint_index completed_updates total_transitions optimizer_steps cumulative_loop_seconds checkpoint_sha256 video poster video_mode".split()
    for result in results:
        rows.append(
            {
                **{k: result[k] for k in metadata_columns},
                **_flatten(result["metrics"], "metrics."),
            }
        )
    (root / "figures").mkdir()
    _plots(root, results)
    with (root / "checkpoint_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row))
        )
        writer.writeheader()
        writer.writerows(rows)
    title = "G1 Flip：四种单仿真与四源联合训练完整实验记录"
    (root / "REPORT.md").write_text(
        f"# {title}\n\n[打开离线视频网页](index.html)\n\n" + body + "\n"
    )
    style = "body{font-family:system-ui,sans-serif;max-width:1450px;margin:2rem auto;padding:0 1rem;line-height:1.6}table{border-collapse:collapse;display:block;overflow-x:auto;font-size:.85rem;margin:1rem 0}th,td{border:1px solid #ccc;padding:.35rem .6rem;white-space:nowrap}th{background:#eee}img,video{max-width:100%;height:auto}details{border:1px solid #ddd;padding:.65rem;margin:1rem 0}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#175bb4}"
    (root / "index.html").write_text(
        f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>{style}</style><body><h1>{title}</h1>{body}</body></html>\n'
    )
    generated = [*(root / name for name in OUTPUTS[:3]), *sorted((root / "figures").iterdir())]
    result = dict(
        complete=True,
        checkpoints=len(results),
        embedded_videos=len(results),
        seed=1,
        protocol=results[0]["protocol"],
        input_sha256=artifacts,
        output_sha256={str(p.relative_to(root)): _hash(p) for p in generated},
    )
    (root / "report_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
