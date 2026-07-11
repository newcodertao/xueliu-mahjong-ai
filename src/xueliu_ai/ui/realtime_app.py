from __future__ import annotations

import json
import queue
import re
import threading
import time
import tkinter as tk
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tkinter import ttk

import cv2

from xueliu_ai.capture.roi_calibrator import calibrate_roi
from xueliu_ai.capture.roi_config import Roi, load_screen_profile, update_roi
from xueliu_ai.capture.screen_capture import ScreenCapture
from xueliu_ai.game_logging.game_logger import GameLogger
from xueliu_ai.mahjong.shanten import best_shanten, normal_shanten, seven_pairs_shanten
from xueliu_ai.mahjong.tiles import TILE_SET
from xueliu_ai.mahjong.ukeire import effective_draws
from xueliu_ai.realtime_table import (
    classify_table_zones,
    classify_table_zones_by_rois,
    diagnose_zones,
    draw_table_overlay,
    reconcile_zone_tile_limits,
    visible_counts_from_zones,
)
from xueliu_ai.strategy.discard_advisor import DiscardAdvice, advise_discard
from xueliu_ai.table.event_classifier import EventTileClassifier
from xueliu_ai.table.game_phase import GamePhase, PhaseContext, should_allow_recommend
from xueliu_ai.table.my_area import analyze_my_area
from xueliu_ai.table.state_fusion import TableStateFusion
from xueliu_ai.table.state_validator import StructuredStateMachine, combine_recommendation_gates
from xueliu_ai.vision.detection_types import Detection
from xueliu_ai.vision.detection_validator import non_max_suppression
from xueliu_ai.vision.yolo_detector import YoloDetector


DEFAULT_MODEL = "models/yolo/xueliu_final325_plus_longjing39_plus83_clean_v1_0709.pt"
PREVIEW_WINDOW = "xueliu realtime preview"
MISSING_SUIT_TO_CODE = {"": None, "万": "W", "筒": "T", "条": "B"}
SUIT_NAMES = {"W": "万", "T": "筒", "B": "条"}


class RealtimeApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("血流麻将实时辅助")
        self.root.geometry("1120x820")

        self.events: queue.Queue[dict[str, object]] = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None

        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.conf_var = tk.DoubleVar(value=0.75)
        self.iou_var = tk.DoubleVar(value=0.45)
        self.imgsz_var = tk.IntVar(value=1280)
        self.interval_var = tk.DoubleVar(value=0.25)
        self.smoothing_frames_var = tk.IntVar(value=5)
        self.missing_suit_var = tk.StringVar(value="")
        self.show_preview_var = tk.BooleanVar(value=True)
        self.save_hard_cases_var = tk.BooleanVar(value=True)
        self.auto_zones_var = tk.BooleanVar(value=True)
        self.manual_zones_var = tk.BooleanVar(value=False)
        self.use_table_context_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="未启动")
        self.phase_var = tk.StringVar(value="-")
        self.recommend_gate_var = tk.StringVar(value="-")
        self.hand_var = tk.StringVar(value="-")
        self.advice_var = tk.StringVar(value="-")
        self.shape_var = tk.StringVar(value="-")
        self.zone_var = tk.StringVar(value="-")
        self.visible_var = tk.StringVar(value="-")
        self.structured_var = tk.StringVar(value="-")
        self.roi_summary_var = tk.StringVar(value="-")

        self._build_layout()
        self._refresh_roi_summary()
        self.root.after(100, self._drain_events)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        controls = ttk.LabelFrame(outer, text="控制", padding=10)
        controls.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(controls, text="模型").pack(anchor=tk.W)
        ttk.Entry(controls, textvariable=self.model_var, width=48).pack(fill=tk.X, pady=(2, 8))

        self._number_row(controls, "置信度", self.conf_var)
        self._number_row(controls, "IOU", self.iou_var)
        self._number_row(controls, "识别尺寸", self.imgsz_var)
        self._number_row(controls, "间隔秒", self.interval_var)
        self._number_row(controls, "稳定帧数", self.smoothing_frames_var)

        ttk.Label(controls, text="定缺").pack(anchor=tk.W, pady=(8, 0))
        ttk.Combobox(
            controls,
            textvariable=self.missing_suit_var,
            values=("", "万", "筒", "条"),
            state="readonly",
            width=12,
        ).pack(anchor=tk.W, pady=(2, 8))

        ttk.Checkbutton(controls, text="显示实时预览窗口", variable=self.show_preview_var).pack(anchor=tk.W)
        ttk.Checkbutton(controls, text="自动保存问题帧", variable=self.save_hard_cases_var).pack(anchor=tk.W)
        ttk.Checkbutton(
            controls,
            text="自动识别手牌/碰杠/弃牌区域",
            variable=self.auto_zones_var,
        ).pack(anchor=tk.W)
        ttk.Checkbutton(
            controls,
            text="使用手动区域覆盖自动分区",
            variable=self.manual_zones_var,
        ).pack(anchor=tk.W)
        ttk.Checkbutton(
            controls,
            text="使用弃牌/碰杠区修正进张",
            variable=self.use_table_context_var,
        ).pack(anchor=tk.W)

        ttk.Separator(controls).pack(fill=tk.X, pady=12)
        ttk.Label(controls, text="开局只需选牌桌区域").pack(anchor=tk.W)
        ttk.Button(controls, text="选择牌桌区域", command=lambda: self._select_roi("table")).pack(fill=tk.X, pady=3)
        ttk.Button(controls, text="推荐-我的底部区域", command=lambda: self._select_roi("my_play_area")).pack(fill=tk.X, pady=3)
        ttk.Button(controls, text="可选-手牌区域", command=lambda: self._select_roi("my_hand")).pack(fill=tk.X, pady=3)

        ttk.Label(controls, text="可选：出现后再补选").pack(anchor=tk.W, pady=(10, 0))
        ttk.Button(controls, text="可选-我的碰杠区", command=lambda: self._select_roi("my_melds")).pack(fill=tk.X, pady=3)
        ttk.Button(controls, text="可选-上家碰杠区", command=lambda: self._select_roi("left_melds")).pack(fill=tk.X, pady=3)
        ttk.Button(controls, text="可选-对家碰杠区", command=lambda: self._select_roi("top_melds")).pack(fill=tk.X, pady=3)
        ttk.Button(controls, text="可选-下家碰杠区", command=lambda: self._select_roi("right_melds")).pack(fill=tk.X, pady=3)
        ttk.Button(controls, text="可选-弃牌区", command=lambda: self._select_roi("discards")).pack(fill=tk.X, pady=3)
        ttk.Button(controls, text="开始识别", command=self._start).pack(fill=tk.X, pady=(14, 3))
        ttk.Button(controls, text="停止识别", command=self._stop).pack(fill=tk.X, pady=3)

        ttk.Label(controls, text="已保存区域").pack(anchor=tk.W, pady=(12, 0))
        ttk.Label(
            controls,
            textvariable=self.roi_summary_var,
            wraplength=330,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, fill=tk.X, pady=(2, 0))

        results = ttk.LabelFrame(outer, text="实时结果", padding=12)
        results.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 0))

        self._result_row(results, "状态", self.status_var)
        self._result_row(results, "牌局阶段", self.phase_var)
        self._result_row(results, "推荐开关", self.recommend_gate_var)
        self._result_row(results, "当前手牌", self.hand_var)
        self._result_row(results, "推荐出牌", self.advice_var)
        self._result_row(results, "牌型/向听", self.shape_var)
        self._result_row(results, "区域统计", self.zone_var)
        self._result_row(results, "可见牌", self.visible_var)

        ttk.Label(results, text="候选建议").pack(anchor=tk.W, pady=(12, 2))
        self._result_row(results, "Structured state", self.structured_var)
        self.candidate_text = tk.Text(results, height=12, wrap=tk.WORD)
        self.candidate_text.pack(fill=tk.BOTH, expand=True)

        ttk.Label(results, text="事件日志").pack(anchor=tk.W, pady=(12, 2))
        self.log_text = tk.Text(results, height=8, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _number_row(self, parent: ttk.Frame, label: str, variable: tk.Variable) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text=label, width=10).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=variable, width=12).pack(side=tk.LEFT)

    def _result_row(self, parent: ttk.Frame, label: str, variable: tk.StringVar) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text=label, width=10).pack(side=tk.LEFT, anchor=tk.N)
        ttk.Label(row, textvariable=variable, wraplength=620, justify=tk.LEFT).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _select_roi(self, name: str) -> None:
        if self.worker and self.worker.is_alive():
            self._append_log("请先停止实时识别，再补选/调整区域。可选区域不影响基础识别。")
            return
        try:
            profile = load_screen_profile()
            roi = calibrate_roi(name=name, monitor=profile.monitor)
            self._refresh_roi_summary()
            self._append_log(f"已保存 {name}: {roi.to_dict()}")
        except Exception as exc:
            self._append_log(f"选择区域失败: {exc}")

    def _start(self) -> None:
        if self.worker and self.worker.is_alive():
            self._append_log("实时识别已经在运行。")
            return
        self.stop_event.clear()
        self.status_var.set("启动中，首次加载模型可能需要 10-20 秒...")
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def _stop(self) -> None:
        self.stop_event.set()
        self.status_var.set("正在停止...")

    def _worker_loop(self) -> None:
        try:
            profile = load_screen_profile()
            capture = ScreenCapture(profile.monitor)
            self.events.put({"type": "status", "message": "正在加载 YOLO 模型..."})
            detector = YoloDetector(self.model_var.get(), image_size=int(self.imgsz_var.get()))
            logger = GameLogger("data/games/realtime_ui.jsonl")
            stable_hand = StableHand(stable_frames=max(1, int(self.smoothing_frames_var.get())))
            region_state = RegionStateMachine()
            structured_state = StructuredStateMachine(minimum_stable_frames=3)
            structural_fusion = TableStateFusion(max_missed=2)
            event_classifier = EventTileClassifier(hu_stable_frames=3, max_missed=2)
            roi_editor = PreviewRoiEditor()
            hard_case_recorder = HardCaseRecorder()
            preview_callback_bound = False
            index = 0
            self.events.put({"type": "status", "message": "运行中"})

            while not self.stop_event.is_set():
                frame = capture.grab().image_bgr
                table_roi = _configured_or_fullscreen(profile.rois.get("table"), frame)
                table_image = table_roi.crop(frame)

                conf = float(self.conf_var.get())
                iou = float(self.iou_var.get())
                detections = _tile_detections(detector.detect_image(table_image, conf=conf, iou=iou), iou)
                if self.manual_zones_var.get():
                    zones = classify_table_zones_by_rois(
                        detections,
                        table_roi,
                        profile.rois,
                        table_image.shape[1],
                        table_image.shape[0],
                    )
                elif self.auto_zones_var.get():
                    zones = classify_table_zones(detections, table_image.shape[1], table_image.shape[0])
                else:
                    zones = classify_table_zones_by_rois(
                        detections,
                        table_roi,
                        profile.rois,
                        table_image.shape[1],
                        table_image.shape[0],
                    )
                low_hand_candidates = _low_confidence_hand_candidates(
                    detector, table_image, conf, iou, enabled=self.auto_zones_var.get()
                )

                hand_tiles = zones.hand
                play_roi = profile.rois.get("my_play_area")
                hand_roi = profile.rois.get("my_hand")
                if not self.auto_zones_var.get() and play_roi and not play_roi.is_empty:
                    play_image = play_roi.crop(frame)
                    play_detections = _tile_detections(detector.detect_image(play_image, conf=conf, iou=iou), iou)
                    hand_tiles, my_meld_tiles = _split_my_play_area(play_detections)
                    zones = _replace_zone_hand_and_melds(zones, hand_tiles, my_meld_tiles)
                elif not self.auto_zones_var.get() and hand_roi and not hand_roi.is_empty:
                    hand_image = hand_roi.crop(frame)
                    hand_detections = _tile_detections(detector.detect_image(hand_image, conf=conf, iou=iou), iou)
                    hand_tiles = [det.label for det in sorted(hand_detections, key=lambda item: item.center_x)]
                    zones = _replace_zone_hand(zones, hand_tiles)

                zones = structural_fusion.update(zones, low_hand_candidates, finalize=False)
                zones = event_classifier.update(zones, table_image.shape[1], table_image.shape[0])

                raw_hand_tiles = hand_tiles
                raw_zones = zones
                zones = reconcile_zone_tile_limits(zones)
                structured_table_state = structural_fusion.build_structured_state(zones)
                zones = structured_table_state.zones
                hand_tiles = zones.hand
                use_table_context = self.use_table_context_var.get()
                diagnostics = diagnose_zones(zones)
                region_check = region_state.update(zones, diagnostics)
                open_melds = diagnostics.open_melds
                stable, visible_hand, stable_message = stable_hand.update(hand_tiles, open_melds)
                missing_suit = MISSING_SUIT_TO_CODE.get(self.missing_suit_var.get(), None)
                decision = should_allow_recommend(
                    PhaseContext(
                        zones=zones,
                        diagnostics=diagnostics,
                        stable=stable,
                        missing_suit=missing_suit,
                        detections=len(detections),
                    )
                )
                if not region_check.valid:
                    decision = type(decision)(
                        phase=decision.phase,
                        allow=False,
                        reasons=[*decision.reasons, f"region_state:{region_check.reason}"],
                    )
                structure_check = structured_state.update(
                    structured_table_state,
                    phase_stable=decision.phase
                    in {GamePhase.WAITING, GamePhase.MY_TURN, GamePhase.PLAYING_PARTIAL},
                    diagnostics_valid=diagnostics.valid,
                )
                if not structure_check.allow_recommend:
                    decision = type(decision)(
                        phase=decision.phase,
                        allow=False,
                        reasons=[*decision.reasons, f"structured_state:{structure_check.reason}"],
                    )
                final_allow = combine_recommendation_gates(
                    structure_check,
                    legacy_state_machine_allow=region_check.valid,
                    phase_allows_recommendation=decision.allow,
                )
                if decision.allow != final_allow:
                    decision = type(decision)(phase=decision.phase, allow=final_allow, reasons=decision.reasons)
                my_area = analyze_my_area(zones)
                visible_counts = (
                    visible_counts_from_zones(zones, include_hand=False) if use_table_context else {}
                )
                advice = (
                    _safe_advice(visible_hand, missing_suit, open_melds, visible_counts)
                    if decision.allow
                    else None
                )
                shape = _shape_summary(visible_hand, open_melds, visible_counts) if visible_hand else stable_message

                payload = {
                    "frame": index,
                    "detections": len(detections),
                    "hand": visible_hand,
                    "raw_hand": raw_hand_tiles,
                    "open_melds": open_melds,
                    "confirmed_open_melds": structured_table_state.confirmed_open_melds,
                    "suspected_open_melds": structured_table_state.suspected_open_melds,
                    "inferred_tile_count": structured_table_state.inferred_tile_count,
                    "structured_state_status": structure_check.state.value,
                    "structured_reason": structure_check.reason,
                    "recommend_block_reason": (
                        None
                        if decision.allow
                        else decision.reasons[0]
                        if decision.reasons
                        else structure_check.reason
                    ),
                    "auto_zones": self.auto_zones_var.get(),
                    "use_table_context": use_table_context,
                    "zones": zones.to_dict(),
                    "raw_zones": raw_zones.to_dict(),
                    "visible_counts": visible_counts,
                    "observed_visible_counts": structured_table_state.observed_visible_counts,
                    "logical_visible_counts": structured_table_state.logical_visible_counts,
                    "phase": decision.phase.value,
                    "phase_text": decision.phase_text,
                    "allow_recommend": decision.allow,
                    "recommend_reasons": decision.reasons,
                    "stable": stable,
                    "stable_message": stable_message,
                    "my_area": my_area.to_dict(),
                    "diagnostics": {
                        "valid": diagnostics.valid,
                        "warnings": diagnostics.warnings,
                        "expected_hand_counts": diagnostics.expected_hand_counts,
                        "open_melds": diagnostics.open_melds,
                        "logical_warnings": diagnostics.logical_warnings,
                    },
                    "region_state": structure_check.to_dict(),
                    "legacy_region_state": region_check.to_dict(),
                    "advice": asdict(advice) if advice else None,
                    "shape": shape,
                    "message": advice.explanation
                    if advice
                    else (decision.reason_text() if not decision.allow else stable_message),
                }
                logger.log("realtime_ui_tick", payload)
                self.events.put({"type": "tick", "payload": payload})

                overlay = None
                if self.save_hard_cases_var.get() and hard_case_recorder.should_save(payload):
                    overlay = _build_preview_overlay(
                        table_image,
                        detections,
                        zones,
                        advice,
                        payload,
                        profile,
                        table_roi,
                        roi_editor,
                    )
                    saved_stem = hard_case_recorder.save(table_image, overlay, payload)
                    if saved_stem:
                        self.events.put({"type": "hard_case", "path": str(saved_stem)})

                if self.show_preview_var.get():
                    if overlay is None:
                        overlay = _build_preview_overlay(
                            table_image,
                            detections,
                            zones,
                            advice,
                            payload,
                            profile,
                            table_roi,
                            roi_editor,
                        )
                    try:
                        if not preview_callback_bound:
                            cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
                            cv2.imshow(PREVIEW_WINDOW, overlay)
                            cv2.waitKey(1)
                            cv2.setMouseCallback(PREVIEW_WINDOW, roi_editor.on_mouse)
                            preview_callback_bound = True
                        else:
                            cv2.imshow(PREVIEW_WINDOW, overlay)
                        key = cv2.waitKey(1) & 0xFF
                        action = roi_editor.handle_key(key)
                        if action == "saved":
                            profile = load_screen_profile()
                            self.manual_zones_var.set(True)
                            self.events.put(
                                {
                                    "type": "status",
                                    "message": f"已保存实时预览手动画区：{roi_editor.last_saved_name}",
                                }
                            )
                        elif key in (27, ord("q")):
                            self.stop_event.set()
                    except cv2.error as exc:
                        self.show_preview_var.set(False)
                        self.events.put(
                            {
                                "type": "status",
                                "message": f"OpenCV 预览窗口不可用，已自动关闭预览：{exc}",
                            }
                        )

                index += 1
                time.sleep(max(0.05, float(self.interval_var.get())))
        except Exception as exc:
            self.events.put({"type": "error", "message": str(exc)})
        finally:
            try:
                cv2.destroyWindow(PREVIEW_WINDOW)
            except cv2.error:
                pass
            self.events.put({"type": "status", "message": "已停止"})

    def _drain_events(self) -> None:
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            if event["type"] == "status":
                self.status_var.set(str(event["message"]))
                self._append_log(str(event["message"]))
            elif event["type"] == "error":
                self.status_var.set("错误")
                self._append_log(f"错误: {event['message']}")
            elif event["type"] == "tick":
                self._render_tick(event["payload"])
            elif event["type"] == "hard_case":
                self._append_log(f"已保存问题帧: {event['path']}")
        self.root.after(100, self._drain_events)

    def _render_tick(self, payload: dict[str, object]) -> None:
        hand = payload.get("hand") or []
        zones = payload.get("zones") or {}
        visible_counts = payload.get("visible_counts") or {}
        advice = payload.get("advice")
        diagnostics = payload.get("diagnostics") or {}
        self.structured_var.set(
            f"{payload.get('structured_state_status', '-')} | "
            f"confirmed {payload.get('confirmed_open_melds', 0)} | "
            f"suspected {payload.get('suspected_open_melds', 0)} | "
            f"inferred {payload.get('inferred_tile_count', 0)} | "
            f"block {payload.get('recommend_block_reason') or '-'}"
        )
        self.phase_var.set(str(payload.get("phase_text") or payload.get("phase") or "-"))
        if payload.get("allow_recommend"):
            self.recommend_gate_var.set("允许推荐")
        else:
            reasons = payload.get("recommend_reasons") or []
            self.recommend_gate_var.set("暂停推荐：" + "；".join(reasons) if reasons else "暂停推荐")
        self.hand_var.set(_tiles_text(hand) if hand else "-")
        self.shape_var.set(str(payload.get("shape") or "-"))
        self.zone_var.set(
            f"手牌 {len(zones.get('hand', []))} / 弃牌 {len(zones.get('center_discards', []))} / "
            f"左 {len(zones.get('left_melds', []))} / 右 {len(zones.get('right_melds', []))} / "
            f"上 {len(zones.get('top_melds', []))} / "
            f"候选副露 {len(zones.get('candidate_meld_tiles', []))} / "
            f"未知 {len(zones.get('unknown_tiles', [])) + len(zones.get('hu_display_tiles', []))} / "
            f"弃牌 我{len(zones.get('my_discards', []))} "
            f"上{len(zones.get('left_discards', []))} "
            f"对{len(zones.get('top_discards', []))} "
            f"下{len(zones.get('right_discards', []))}"
        )
        unknown_like = (
            len(zones.get("unknown_tiles", []))
            + len(zones.get("candidate_meld_tiles", []))
            + len(zones.get("hu_display_tiles", []))
            + len(zones.get("event_tiles", []))
        )
        self.zone_var.set(
            f"hand {len(zones.get('hand', []))} / discards {len(zones.get('center_discards', []))} / "
            f"L {len(zones.get('left_melds', []))} / R {len(zones.get('right_melds', []))} / "
            f"T {len(zones.get('top_melds', []))} / unknown {unknown_like} / "
            f"discard me {len(zones.get('my_discards', []))} "
            f"L {len(zones.get('left_discards', []))} "
            f"T {len(zones.get('top_discards', []))} "
            f"R {len(zones.get('right_discards', []))}"
        )
        self.visible_var.set(_counts_text(visible_counts) if visible_counts else "-")
        if diagnostics and not diagnostics.get("valid", True):
            self.status_var.set("区域异常，暂停推荐")

        if advice:
            self.advice_var.set(_humanize_text(str(advice["explanation"])))
            candidates = advice.get("candidates", [])[:8]
            text = "\n".join(
                f"{_tile_text(item['tile'])}: 分数 {item['score']:.1f}, 向听 {item['shanten']}, 进张 {item['ukeire']}"
                for item in candidates
            )
        else:
            self.advice_var.set(str(payload.get("message") or "-"))
            text = "-"
        self.candidate_text.delete("1.0", tk.END)
        self.candidate_text.insert(tk.END, text)

    def _append_log(self, message: str) -> None:
        self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} {message}\n")
        self.log_text.see(tk.END)

    def _refresh_roi_summary(self) -> None:
        try:
            profile = load_screen_profile()
        except Exception as exc:
            self.roi_summary_var.set(f"读取失败：{exc}")
            return
        names = [
            ("table", "牌桌"),
            ("my_play_area", "我的底部区域"),
            ("my_hand", "手牌"),
            ("my_melds", "我的碰杠"),
            ("left_melds", "上家碰杠"),
            ("top_melds", "对家碰杠"),
            ("right_melds", "下家碰杠"),
            ("discards", "弃牌"),
        ]
        lines = []
        for key, label in names:
            roi = profile.rois.get(key)
            if roi and not roi.is_empty:
                lines.append(f"{label}: {roi.x},{roi.y},{roi.width}x{roi.height}")
        self.roi_summary_var.set("\n".join(lines) if lines else "暂无；首次只需要框选牌桌区域")

    def _on_close(self) -> None:
        self.stop_event.set()
        self.root.after(150, self.root.destroy)


@dataclass
class StableHand:
    stable_frames: int = 2
    history: deque[tuple[str, ...]] = field(default_factory=deque)

    def update(self, tiles: list[str], open_melds: int = 0) -> tuple[bool, list[str], str]:
        expected_counts = {13 - open_melds * 3, 14 - open_melds * 3}
        if len(tiles) not in expected_counts:
            self.history.clear()
            expected_text = "/".join(str(value) for value in sorted(expected_counts))
            return False, tiles, f"已开门 {open_melds} 组，暗手牌应为 {expected_text} 张，当前 {len(tiles)} 张"
        over = [tile for tile, count in Counter(tiles).items() if count > 4]
        if over:
            self.history.clear()
            return False, tiles, f"牌数量异常：{_tiles_text(over)} 超过 4 张"
        current = tuple(tiles)
        self.history.append(current)
        while len(self.history) > self.stable_frames:
            self.history.popleft()
        if len(self.history) < self.stable_frames:
            return False, tiles, "等待连续稳定帧"
        if len(set(self.history)) != 1:
            return False, tiles, "连续帧还不稳定"
        return True, tiles, "稳定"


@dataclass(frozen=True)
class RegionStateResult:
    valid: bool
    reason: str = "ok"
    previous_open_melds: int = 0
    current_open_melds: int = 0
    unknown_like_tiles: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "reason": self.reason,
            "previous_open_melds": self.previous_open_melds,
            "current_open_melds": self.current_open_melds,
            "unknown_like_tiles": self.unknown_like_tiles,
        }


@dataclass
class RegionStateMachine:
    last_open_melds: int | None = None
    last_hand_count: int | None = None

    def update(self, zones, diagnostics) -> RegionStateResult:
        current_open_melds = int(diagnostics.open_melds)
        unknown_like = (
            len(zones.unknown_tiles)
            + len(zones.candidate_meld_tiles)
            + len(zones.hu_display_tiles)
            + len(getattr(zones, "event_tiles", []))
        )
        previous = self.last_open_melds if self.last_open_melds is not None else current_open_melds

        if zones.event_tiles:
            return RegionStateResult(False, "event_animation_active", previous, current_open_melds, unknown_like)
        if not diagnostics.valid:
            return RegionStateResult(False, "diagnostics_invalid", previous, current_open_melds, unknown_like)
        if unknown_like >= 3:
            return RegionStateResult(False, "too_many_unknown_or_event_tiles", previous, current_open_melds, unknown_like)
        if self.last_open_melds is not None and current_open_melds - self.last_open_melds > 1:
            return RegionStateResult(False, "illegal_meld_jump", previous, current_open_melds, unknown_like)
        if len(zones.hand) not in set(diagnostics.expected_hand_counts):
            return RegionStateResult(False, "illegal_hand_count", previous, current_open_melds, unknown_like)

        self.last_open_melds = current_open_melds
        self.last_hand_count = len(zones.hand)
        return RegionStateResult(True, "ok", previous, current_open_melds, unknown_like)


@dataclass
class RealtimeStateStabilizer:
    window_size: int = 5
    zone_history: deque[dict[str, object]] = field(default_factory=deque)
    hand_history: deque[list[str]] = field(default_factory=deque)
    last_zones: object | None = None
    last_hand: list[str] = field(default_factory=list)

    def update(self, zones, hand_tiles: list[str]):
        if self.last_hand and _is_implausible_hand_jump(self.last_hand, hand_tiles):
            return self.last_zones or zones, self.last_hand

        self.zone_history.append(zones.to_dict())
        self.hand_history.append(list(hand_tiles))
        while len(self.zone_history) > self.window_size:
            self.zone_history.popleft()
        while len(self.hand_history) > self.window_size:
            self.hand_history.popleft()

        stable_hand = _stabilize_ordered_tiles(list(self.hand_history), target_len_hint=len(hand_tiles))
        stable_zones = _stabilize_zones(list(self.zone_history), zones)
        if stable_hand:
            stable_zones = type(zones)(
                hand=stable_hand,
                bottom_melds=stable_zones.bottom_melds,
                left_melds=stable_zones.left_melds,
                right_melds=stable_zones.right_melds,
                top_melds=stable_zones.top_melds,
                center_discards=stable_zones.center_discards,
                all_tiles=stable_zones.all_tiles,
                my_discards=stable_zones.my_discards,
                left_discards=stable_zones.left_discards,
                top_discards=stable_zones.top_discards,
                right_discards=stable_zones.right_discards,
                unknown_tiles=stable_zones.unknown_tiles,
                candidate_meld_tiles=stable_zones.candidate_meld_tiles,
                hu_display_tiles=stable_zones.hu_display_tiles,
                event_tiles=stable_zones.event_tiles,
                table_decor_tiles=stable_zones.table_decor_tiles,
                zone_tiles=zones.zone_tiles,
                meld_groups=zones.meld_groups,
            )
        output_hand = stable_hand or hand_tiles
        self.last_zones = stable_zones
        self.last_hand = list(output_hand)
        return stable_zones, output_hand


def launch_realtime_app() -> None:
    root = tk.Tk()
    RealtimeApp(root)
    root.mainloop()


def _configured_or_fullscreen(roi: Roi | None, frame) -> Roi:
    if roi and not roi.is_empty:
        return roi
    height, width = frame.shape[:2]
    return Roi(0, 0, width, height)


def _preview_status_text(payload: dict[str, object], advice: DiscardAdvice | None) -> str:
    phase = str(payload.get("phase") or "-")
    allow = "allow" if payload.get("allow_recommend") else "blocked"
    if advice:
        return f"phase={phase}; recommend={advice.recommended}"
    return f"phase={phase}; recommend={allow}; see main UI"


def _build_preview_overlay(
    table_image,
    detections: list[Detection],
    zones,
    advice: DiscardAdvice | None,
    payload: dict[str, object],
    profile,
    table_roi: Roi,
    roi_editor: "PreviewRoiEditor",
):
    overlay = draw_table_overlay(
        table_image,
        detections,
        zones,
        _tile_text(advice.recommended) if advice else None,
        _preview_status_text(payload, advice),
    )
    _draw_configured_rois(overlay, table_roi, profile.rois)
    roi_editor.draw(overlay)
    return overlay


class HardCaseRecorder:
    def __init__(
        self,
        root: str | Path = "data/hard_cases/realtime_missed",
        min_interval_seconds: float = 3.0,
    ) -> None:
        self.root = Path(root)
        self.min_interval_seconds = min_interval_seconds
        self.last_saved_at_by_reason: dict[str, float] = {}

    def should_save(self, payload: dict[str, object]) -> bool:
        reasons = self._reasons(payload)
        if not reasons:
            return False
        key = "|".join(reasons)
        now = time.time()
        last = self.last_saved_at_by_reason.get(key, 0.0)
        if now - last < self.min_interval_seconds:
            return False
        self.last_saved_at_by_reason[key] = now
        payload["hard_case_reasons"] = reasons
        return True

    def save(self, raw_image, overlay_image, payload: dict[str, object]) -> Path | None:
        reasons = payload.get("hard_case_reasons") or self._reasons(payload)
        if not reasons:
            return None
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        frame = int(payload.get("frame") or 0)
        day_dir = self.root / time.strftime("%Y%m%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        reason_slug = _slugify("_".join(str(reason) for reason in reasons))[:80]
        stem = day_dir / f"{timestamp}_frame{frame:06d}_{reason_slug}"
        cv2.imwrite(str(stem.with_name(stem.name + "_raw.jpg")), raw_image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        cv2.imwrite(str(stem.with_name(stem.name + "_overlay.jpg")), overlay_image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        stem.with_suffix(".json").write_text(
            json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return stem

    def _reasons(self, payload: dict[str, object]) -> list[str]:
        reasons: list[str] = []
        diagnostics = payload.get("diagnostics") or {}
        if isinstance(diagnostics, dict) and not diagnostics.get("valid", True):
            reasons.append("zone_invalid")
            for warning in diagnostics.get("warnings", [])[:3]:
                reasons.append(_slugify(str(warning))[:32])
        hand = payload.get("hand") or []
        raw_hand = payload.get("raw_hand") or []
        expected = diagnostics.get("expected_hand_counts", []) if isinstance(diagnostics, dict) else []
        if expected and len(hand) not in set(expected):
            reasons.append(f"hand_count_{len(hand)}_expected_{'-'.join(map(str, expected))}")
        if expected and len(raw_hand) not in set(expected):
            reasons.append(f"raw_hand_count_{len(raw_hand)}")
        all_tiles = []
        zones = payload.get("zones") or {}
        if isinstance(zones, dict):
            for name in ("hand", "bottom_melds", "left_melds", "right_melds", "top_melds", "center_discards"):
                all_tiles.extend(zones.get(name, []) or [])
            unknown_like = (
                len(zones.get("unknown_tiles", []) or [])
                + len(zones.get("candidate_meld_tiles", []) or [])
                + len(zones.get("hu_display_tiles", []) or [])
                + len(zones.get("event_tiles", []) or [])
            )
            if unknown_like:
                reasons.append(f"unknown_event_tiles_{unknown_like}")
        region_state = payload.get("region_state") or {}
        if isinstance(region_state, dict) and not region_state.get("valid", True):
            reasons.append("region_state_" + _slugify(str(region_state.get("reason") or "invalid"))[:40])
        over_limit = [tile for tile, count in Counter(all_tiles).items() if count > 4]
        if over_limit:
            reasons.append("tile_over_4_" + "-".join(sorted(over_limit)))
        if not payload.get("stable", True):
            stable_message = _slugify(str(payload.get("stable_message") or "unstable"))
            reasons.append(f"unstable_{stable_message[:40]}")
        blocked_reasons = [str(reason) for reason in (payload.get("recommend_reasons") or [])]
        if any(any(keyword in reason for keyword in ("区域", "手牌", "数量", "稳定")) for reason in blocked_reasons):
            reasons.append("recommend_blocked_by_recognition")
        return list(dict.fromkeys(reasons))


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _slugify(text: str) -> str:
    text = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "case"


class PreviewRoiEditor:
    shortcuts = {
        ord("h"): ("my_hand", "HAND"),
        ord("m"): ("my_melds", "MY MELD"),
        ord("l"): ("left_melds", "LEFT MELD"),
        ord("t"): ("top_melds", "TOP MELD"),
        ord("r"): ("right_melds", "RIGHT MELD"),
        ord("d"): ("discards", "DISCARD"),
        ord("p"): ("my_play_area", "MY PLAY AREA"),
    }

    colors = {
        "my_hand": (0, 220, 255),
        "my_melds": (0, 220, 80),
        "left_melds": (0, 180, 80),
        "top_melds": (0, 180, 80),
        "right_melds": (0, 180, 80),
        "discards": (255, 180, 0),
        "my_play_area": (220, 220, 0),
    }

    def __init__(self) -> None:
        self.active_name: str | None = None
        self.active_label = ""
        self.dragging = False
        self.start: tuple[int, int] | None = None
        self.current: tuple[int, int] | None = None
        self.pending_rect: tuple[int, int, int, int] | None = None
        self.last_saved_name = ""

    def handle_key(self, key: int) -> str | None:
        if key in self.shortcuts:
            self.active_name, self.active_label = self.shortcuts[key]
            self.pending_rect = None
            return "selected"
        if key in (ord("s"), 13, 32) and self.active_name and self.pending_rect:
            x1, y1, x2, y2 = self.pending_rect
            if x2 - x1 >= 5 and y2 - y1 >= 5:
                self._save_crop_roi(self.active_name, x1, y1, x2, y2)
                self.last_saved_name = self.active_name
                self.pending_rect = None
                return "saved"
        if key in (ord("x"), ord("c")):
            self.pending_rect = None
            self.dragging = False
            return "cancelled"
        return None

    def on_mouse(self, event: int, x: int, y: int, flags: int, table_roi: Roi) -> None:
        if not self.active_name:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start = (x, y)
            self.current = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging and self.start:
            self.dragging = False
            self.current = (x, y)
            x1, y1 = self.start
            self.pending_rect = (
                min(x1, x),
                min(y1, y),
                max(x1, x),
                max(y1, y),
            )

    def draw(self, canvas) -> None:
        lines = [
            "ROI edit: H hand, M my_meld, L/T/R melds, D discard, P my_area",
            "drag mouse, then S/Enter/Space save; C cancel",
        ]
        if self.active_name:
            lines.append(f"editing: {self.active_label}")
        _draw_small_help(canvas, lines)

        rect = self.pending_rect
        if self.dragging and self.start and self.current:
            x1, y1 = self.start
            x2, y2 = self.current
            rect = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        if rect and self.active_name:
            color = self.colors.get(self.active_name, (255, 255, 255))
            x1, y1, x2, y2 = rect
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(canvas, self.active_label, (x1 + 4, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    def _save_crop_roi(self, name: str, x1: int, y1: int, x2: int, y2: int) -> None:
        # The callback receives crop coordinates; cv2 passes table_roi as userdata.
        # It is read from the current window callback userdata via _last_table_roi.
        table_roi = _CURRENT_TABLE_ROI
        update_roi(
            name,
            Roi(
                x=table_roi.x + int(x1),
                y=table_roi.y + int(y1),
                width=int(x2 - x1),
                height=int(y2 - y1),
            ),
        )


_CURRENT_TABLE_ROI = Roi(0, 0, 0, 0)


def _draw_configured_rois(canvas, table_roi: Roi, rois: dict[str, Roi]) -> None:
    global _CURRENT_TABLE_ROI
    _CURRENT_TABLE_ROI = table_roi
    labels = {
        "my_hand": ("HAND ROI", (0, 220, 255)),
        "my_melds": ("MY MELD ROI", (0, 220, 80)),
        "left_melds": ("LEFT ROI", (0, 180, 80)),
        "top_melds": ("TOP ROI", (0, 180, 80)),
        "right_melds": ("RIGHT ROI", (0, 180, 80)),
        "discards": ("DISCARD ROI", (255, 180, 0)),
        "my_play_area": ("MY AREA ROI", (220, 220, 0)),
    }
    for name, (label, color) in labels.items():
        roi = rois.get(name)
        if not roi or roi.is_empty:
            continue
        x1 = roi.x - table_roi.x
        y1 = roi.y - table_roi.y
        x2 = x1 + roi.width
        y2 = y1 + roi.height
        if x2 < 0 or y2 < 0 or x1 >= canvas.shape[1] or y1 >= canvas.shape[0]:
            continue
        x1 = max(0, min(canvas.shape[1] - 1, x1))
        y1 = max(0, min(canvas.shape[0] - 1, y1))
        x2 = max(0, min(canvas.shape[1] - 1, x2))
        y2 = max(0, min(canvas.shape[0] - 1, y2))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, label, (x1 + 4, max(18, y1 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def _draw_small_help(canvas, lines: list[str]) -> None:
    x, y = 8, canvas.shape[0] - 54
    width = min(canvas.shape[1] - 16, 760)
    height = 48 + max(0, len(lines) - 2) * 18
    cv2.rectangle(canvas, (x - 2, y - 20), (x + width, y + height - 18), (0, 0, 0), -1)
    for index, line in enumerate(lines):
        cv2.putText(canvas, line, (x, y + index * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)


def _replace_zone_hand(zones, hand_tiles: list[str]):
    return type(zones)(
        hand=hand_tiles,
        bottom_melds=zones.bottom_melds,
        left_melds=zones.left_melds,
        right_melds=zones.right_melds,
        top_melds=zones.top_melds,
        center_discards=zones.center_discards,
        all_tiles=zones.all_tiles,
        my_discards=zones.my_discards,
        left_discards=zones.left_discards,
        top_discards=zones.top_discards,
        right_discards=zones.right_discards,
        unknown_tiles=zones.unknown_tiles,
        candidate_meld_tiles=zones.candidate_meld_tiles,
        hu_display_tiles=zones.hu_display_tiles,
        event_tiles=zones.event_tiles,
        table_decor_tiles=zones.table_decor_tiles,
        zone_tiles=zones.zone_tiles,
        meld_groups=zones.meld_groups,
    )


def _replace_zone_hand_and_melds(zones, hand_tiles: list[str], my_meld_tiles: list[str]):
    return type(zones)(
        hand=hand_tiles,
        bottom_melds=my_meld_tiles,
        left_melds=zones.left_melds,
        right_melds=zones.right_melds,
        top_melds=zones.top_melds,
        center_discards=zones.center_discards,
        all_tiles=zones.all_tiles,
        my_discards=zones.my_discards,
        left_discards=zones.left_discards,
        top_discards=zones.top_discards,
        right_discards=zones.right_discards,
        unknown_tiles=zones.unknown_tiles,
        candidate_meld_tiles=zones.candidate_meld_tiles,
        hu_display_tiles=zones.hu_display_tiles,
        event_tiles=zones.event_tiles,
        table_decor_tiles=zones.table_decor_tiles,
        zone_tiles=zones.zone_tiles,
        meld_groups=zones.meld_groups,
    )


def _recover_auto_hand_with_lower_confidence(
    zones,
    detector: YoloDetector,
    table_image,
    conf: float,
    iou: float,
    enabled: bool,
):
    if not enabled:
        return zones
    recovery_conf = max(0.55, conf - 0.18)
    if recovery_conf >= conf:
        return zones
    low_detections = _tile_detections(detector.detect_image(table_image, conf=recovery_conf, iou=iou), iou)
    low_zones = classify_table_zones(low_detections, table_image.shape[1], table_image.shape[0])
    if not _should_use_recovered_hand(zones.hand, low_zones.hand):
        return zones
    return _replace_zone_hand(zones, low_zones.hand)


def _low_confidence_hand_candidates(
    detector: YoloDetector,
    table_image,
    conf: float,
    iou: float,
    enabled: bool,
):
    if not enabled:
        return []
    recovery_conf = max(0.55, conf - 0.18)
    if recovery_conf >= conf:
        return []
    detections = _tile_detections(detector.detect_image(table_image, conf=recovery_conf, iou=iou), iou)
    zones = classify_table_zones(detections, table_image.shape[1], table_image.shape[0])
    return [tile for tile in zones.zone_tiles if tile.zone == "hand" and tile.confidence < conf]


def _should_use_recovered_hand(current_hand: list[str], recovered_hand: list[str]) -> bool:
    valid_counts = {1, 2, 4, 5, 7, 8, 10, 11, 13, 14}
    if len(recovered_hand) not in valid_counts:
        return False
    if len(recovered_hand) < len(current_hand):
        return False
    if any(tile not in TILE_SET for tile in recovered_hand):
        return False
    if any(count > 4 for count in Counter(recovered_hand).values()):
        return False
    if len(recovered_hand) == len(current_hand):
        return _is_plausible_same_hand_recovery(current_hand, recovered_hand)
    return len(recovered_hand) == len(current_hand) + 1


def _is_plausible_same_hand_recovery(current_hand: list[str], recovered_hand: list[str]) -> bool:
    current_counts = Counter(current_hand)
    recovered_counts = Counter(recovered_hand)
    removed = sum((current_counts - recovered_counts).values())
    added = sum((recovered_counts - current_counts).values())
    return removed <= 1 and added <= 1


def _tile_detections(detections: list[Detection], iou: float) -> list[Detection]:
    return non_max_suppression([det for det in detections if det.label in TILE_SET], iou)


def _split_my_play_area(detections: list[Detection]) -> tuple[list[str], list[str]]:
    if not detections:
        return [], []
    hand_detections = _bottom_hand_tile_run(detections)
    hand_ids = {id(det) for det in hand_detections}
    meld_detections = [det for det in detections if id(det) not in hand_ids]
    hand_tiles = [det.label for det in sorted(hand_detections, key=lambda item: item.center_x)]
    meld_tiles = [det.label for det in sorted(meld_detections, key=lambda item: (item.y1, item.center_x))]
    return hand_tiles, meld_tiles


def _bottom_hand_tile_run(detections: list[Detection]) -> list[Detection]:
    candidates = _horizontal_tile_runs(detections)
    if not candidates:
        return []
    valid_hand_counts = {1, 2, 4, 5, 7, 8, 10, 11, 13, 14}
    valid_candidates = [group for group in candidates if len(group) in valid_hand_counts]
    if valid_candidates:
        return max(
            valid_candidates,
            key=lambda group: (
                _group_center_y(group),
                len(group),
                sum(item.area for item in group),
            ),
        )
    return max(
        candidates,
        key=lambda group: (
            len(group),
            sum(item.area for item in group),
            _group_center_y(group),
        ),
    )


def _horizontal_tile_runs(detections: list[Detection]) -> list[list[Detection]]:
    if len(detections) <= 2:
        return [sorted(detections, key=lambda item: item.center_x)]
    median_height = sorted(max(1.0, det.y2 - det.y1) for det in detections)[len(detections) // 2]
    rows: list[list[Detection]] = []
    for det in sorted(detections, key=lambda item: (item.y1 + item.y2) / 2):
        center_y = (det.y1 + det.y2) / 2
        for row in rows:
            row_center = sum((item.y1 + item.y2) / 2 for item in row) / len(row)
            if abs(center_y - row_center) <= median_height * 0.45:
                row.append(det)
                break
        else:
            rows.append([det])

    candidates: list[list[Detection]] = []
    for row in rows:
        ordered = sorted(row, key=lambda item: item.center_x)
        if len(ordered) <= 1:
            candidates.append(ordered)
            continue
        median_width = sorted(max(1.0, det.x2 - det.x1) for det in ordered)[len(ordered) // 2]
        max_gap = median_width * 1.9
        current: list[Detection] = []
        for det in ordered:
            if not current or det.center_x - current[-1].center_x <= max_gap:
                current.append(det)
            else:
                candidates.append(current)
                current = [det]
        if current:
            candidates.append(current)

    return candidates


def _group_center_y(detections: list[Detection]) -> float:
    if not detections:
        return 0.0
    return sum((item.y1 + item.y2) / 2 for item in detections) / len(detections)


def _safe_advice(
    tiles: list[str],
    missing_suit: str | None,
    open_melds: int = 0,
    visible_counts: dict[str, int] | None = None,
) -> DiscardAdvice | None:
    if len(tiles) != 14 - open_melds * 3:
        return None
    return advise_discard(tiles, missing_suit, visible_counts=visible_counts, open_melds=open_melds)


def _shape_summary(
    tiles: list[str],
    open_melds: int = 0,
    visible_counts: dict[str, int] | None = None,
) -> str:
    expected_counts = {13 - open_melds * 3, 14 - open_melds * 3}
    if len(tiles) not in expected_counts:
        expected_text = "/".join(str(value) for value in sorted(expected_counts))
        return f"已开门 {open_melds} 组，等待暗手牌 {expected_text} 张；当前 {len(tiles)} 张"
    normal = normal_shanten(tiles, open_melds=open_melds)
    seven_pairs = seven_pairs_shanten(tiles) if open_melds == 0 else None
    best = best_shanten(tiles, open_melds=open_melds)
    draws = effective_draws(tiles, visible_counts=visible_counts, open_melds=open_melds)
    draw_text = " ".join(f"{_tile_text(tile)}x{count}" for tile, count in list(draws.items())[:12])
    if not draw_text:
        draw_text = "无"
    seven_pairs_text = f"，七对 {seven_pairs}" if seven_pairs is not None else ""
    return f"已开门 {open_melds} 组；最佳向听 {best}；普通 {normal}{seven_pairs_text}；有效牌 {draw_text}"


def _open_melds_from_concealed_count(count: int) -> int:
    if count in (13, 14):
        return 0
    if count in (10, 11):
        return 1
    if count in (7, 8):
        return 2
    if count in (4, 5):
        return 3
    if count in (1, 2):
        return 4
    return 0


def _open_melds_from_visible_tiles(tiles: list[str]) -> int:
    if not tiles:
        return 0
    return min(4, round(len(tiles) / 3))


def _tile_text(tile: str) -> str:
    value = str(tile).strip().upper()
    if len(value) == 2 and value[0].isdigit() and value[1] in SUIT_NAMES:
        return f"{value[0]}{SUIT_NAMES[value[1]]}"
    return str(tile)


def _tiles_text(tiles: list[str]) -> str:
    return " ".join(_tile_text(tile) for tile in tiles)


def _counts_text(counts: dict[str, int]) -> str:
    ordered = sorted(counts.items(), key=lambda item: (item[0][-1], int(item[0][0])))
    return " ".join(f"{_tile_text(tile)}x{count}" for tile, count in ordered)


def _humanize_text(text: str) -> str:
    return re.sub(r"\b([1-9][WBT])\b", lambda match: _tile_text(match.group(1)), text)


def _stabilize_ordered_tiles(history: list[list[str]], target_len_hint: int) -> list[str]:
    if not history:
        return []
    target_len = _majority_length(history, target_len_hint)
    if target_len <= 0:
        return []
    votes: list[Counter[str]] = [Counter() for _ in range(target_len)]
    for tiles in history:
        aligned = _align_tiles_to_length(tiles, target_len)
        for index, tile in enumerate(aligned):
            if tile:
                votes[index][tile] += 1

    threshold = max(1, len(history) // 2)
    result: list[str] = []
    for counter in votes:
        if not counter:
            return []
        tile, count = counter.most_common(1)[0]
        if count < threshold:
            return []
        result.append(tile)
    return result


def _is_implausible_hand_jump(previous: list[str], current: list[str]) -> bool:
    valid_counts = {1, 2, 4, 5, 7, 8, 10, 11, 13, 14}
    if len(current) not in valid_counts:
        return True
    return abs(len(previous) - len(current)) > 4


def _stabilize_zones(history: list[dict[str, list[str]]], latest_zones):
    if not history:
        return latest_zones
    zone_names = [
        "hand",
        "bottom_melds",
        "left_melds",
        "right_melds",
        "top_melds",
        "center_discards",
        "all_tiles",
        "my_discards",
        "left_discards",
        "top_discards",
        "right_discards",
        "unknown_tiles",
        "candidate_meld_tiles",
        "hu_display_tiles",
        "event_tiles",
        "table_decor_tiles",
    ]
    values = {}
    for name in zone_names:
        zone_history = [list(row.get(name, [])) for row in history]
        if name == "hand":
            values[name] = _stabilize_ordered_tiles(zone_history, len(getattr(latest_zones, name)))
        else:
            values[name] = _stabilize_multiset(zone_history)
    values["zone_tiles"] = latest_zones.zone_tiles
    values["meld_groups"] = latest_zones.meld_groups
    return type(latest_zones)(**values)


def _stabilize_multiset(history: list[list[str]]) -> list[str]:
    if not history:
        return []
    max_counts: Counter[str] = Counter()
    for tiles in history:
        counts = Counter(tiles)
        for tile, count in counts.items():
            max_counts[tile] = max(max_counts[tile], count)

    threshold = max(1, len(history) // 2)
    result: list[str] = []
    for tile in sorted(max_counts, key=lambda value: (value[-1], int(value[0]))):
        for occurrence in range(max_counts[tile]):
            present = sum(1 for tiles in history if Counter(tiles)[tile] > occurrence)
            if present >= threshold:
                result.append(tile)
    return result


def _majority_length(history: list[list[str]], fallback: int) -> int:
    lengths = Counter(len(tiles) for tiles in history)
    if not lengths:
        return fallback
    length, count = lengths.most_common(1)[0]
    if count >= max(1, len(history) // 2):
        return length
    return fallback


def _align_tiles_to_length(tiles: list[str], target_len: int) -> list[str | None]:
    if len(tiles) == target_len:
        return list(tiles)
    if len(tiles) > target_len:
        return list(tiles[:target_len])
    missing = target_len - len(tiles)
    if missing <= 0:
        return list(tiles)
    left_pad = missing // 2
    right_pad = missing - left_pad
    return [None] * left_pad + list(tiles) + [None] * right_pad
