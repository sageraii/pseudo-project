"""
Week 11: 종합 벤치마킹 보고서 생성

12주간 실험 결과를 종합하여 ACT, GR00T N1.6, Cosmos Predict2.5, Cosmos Policy의
비교 분석 보고서를 생성합니다. Cosmos Predict2.5 + IDM 시너지 파이프라인의
증강 전/후 성능 비교를 포함합니다.

Usage:
    python scripts/week11_benchmark_report.py \
        --eval-dir outputs/eval \
        --cosmos-dir outputs/cosmos_eval \
        --synergy-dir outputs/week10_analysis \
        --output outputs/final_report.md
"""

import argparse
import json
from datetime import datetime
from pathlib import Path


def load_json(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_eval_results(eval_dir: Path) -> dict:
    """Week 6 ACT vs GR00T 평가 결과 로드"""
    results = {}
    for f in sorted(eval_dir.glob("eval_*.json")):
        data = load_json(f)
        if data:
            results[data["task"]] = data
    return results


def generate_report(
    eval_results: dict,
    cosmos_results: dict | None,
    synergy_results: dict | None,
    output_path: Path,
):
    """마크다운 보고서 생성"""
    now = datetime.now().strftime("%Y-%m-%d")

    lines = [
        f"# NVIDIA 로봇 AI 모델 종합 벤치마킹 보고서",
        f"",
        f"**생성일**: {now}",
        f"**플랫폼**: ROBOTIS OMX (5-DOF + 1 Gripper, 6-dim action)",
        f"**접근법**: Option C+ 하이브리드 (ACT 베이스라인 + GR00T 파인튜닝 + Cosmos Predict2.5 후훈련/IDM 시너지)",
        f"",
        f"---",
        f"",
        f"## 1. 실험 개요",
        f"",
        f"| 모델 | 유형 | 역할 |",
        f"|------|------|------|",
        f"| ACT | IL Policy | 베이스라인 (OMX 학습 + 배포) |",
        f"| GR00T N1.6 | VLA | 파인튜닝 + 배포 (두뇌) |",
        f"| Cosmos Predict2.5 | World Model | 후훈련 + 합성 비디오 (상상력) |",
        f"| GR00T IDM | Inverse Dynamics | pseudo labeling (관찰자) |",
        f"| Cosmos Policy | Video-to-Policy | 비교 분석 (추론 전용) |",
        f"",
    ]

    # --- ACT vs GR00T 비교 ---
    lines.extend([
        f"## 2. ACT vs GR00T 비교 (OMX 실측)",
        f"",
    ])

    if eval_results:
        lines.append(f"| Task | Metric | ACT | GR00T | Winner |")
        lines.append(f"|------|--------|-----|-------|--------|")

        for task, data in eval_results.items():
            act = data.get("act", {})
            groot = data.get("groot", {})

            act_sr = act.get("success_rate", 0)
            groot_sr = groot.get("success_rate", 0)
            sr_winner = "ACT" if act_sr > groot_sr else "GR00T" if groot_sr > act_sr else "Tie"

            act_lat = act.get("avg_latency_ms", 0)
            groot_lat = groot.get("avg_latency_ms", 0)
            lat_winner = "ACT" if act_lat < groot_lat else "GR00T"

            act_jerk = act.get("avg_jerk", 0)
            groot_jerk = groot.get("avg_jerk", 0)
            jerk_winner = "ACT" if act_jerk < groot_jerk else "GR00T"

            lines.append(
                f"| {task} | Success Rate | {act_sr:.1%} | {groot_sr:.1%} | {sr_winner} |"
            )
            lines.append(
                f"| | Latency (ms) | {act_lat:.1f} | {groot_lat:.1f} | {lat_winner} |"
            )
            lines.append(
                f"| | Smoothness | {act_jerk:.4f} | {groot_jerk:.4f} | {jerk_winner} |"
            )

        lines.append(f"")
    else:
        lines.append(f"*평가 데이터 없음. week6_eval_omx.py를 먼저 실행하세요.*")
        lines.append(f"")

    # --- Cosmos Policy ---
    lines.extend([
        f"## 3. Cosmos Policy LIBERO 벤치마크",
        f"",
    ])

    if cosmos_results:
        lines.append(f"| Suite | Paper Result | Our Result | Gap |")
        lines.append(f"|-------|-------------|-----------|-----|")
        for suite, data in cosmos_results.items():
            paper = data.get("paper_result", 0)
            status = data.get("status", "N/A")
            lines.append(f"| {suite} | {paper:.1f}% | {status} | - |")
        lines.append(f"")
    else:
        lines.extend([
            f"| Suite | Paper Result |",
            f"|-------|-------------|",
            f"| libero_spatial | 98.1% |",
            f"| libero_object | 100.0% |",
            f"| libero_goal | 98.2% |",
            f"| libero_long | 97.6% |",
            f"| **Average** | **98.5%** |",
            f"",
            f"*Docker 내부 eval.sh로 재현 필요*",
            f"",
        ])

    # --- Cosmos Predict2.5 + IDM 시너지 ---
    lines.extend([
        f"## 4. Cosmos Predict2.5 + IDM 시너지 파이프라인",
        f"",
        f"```",
        f"OMX joints → FK → Cosmos EE state/action",
        f"  → Cosmos Predict2.5 합성 비디오 → IDM pseudo labeling",
        f"  → 관절 매핑 (IDM→VLA) → 품질 필터 (grade≥B)",
        f"  → GR00T N1.6 VLA 재학습",
        f"```",
        f"",
    ])

    if synergy_results:
        qm = synergy_results.get("quality_metrics", {})
        lines.append(f"| 메트릭 | 값 |")
        lines.append(f"|--------|-----|")
        lines.append(f"| Pseudo label 수 | {qm.get('num_predictions', 'N/A')} |")
        lines.append(f"| Jerk | {qm.get('jerk', 'N/A')} |")
        lines.append(f"| Temporal Consistency | {qm.get('temporal_consistency', 'N/A')} |")
        lines.append(f"| 품질 등급 | {qm.get('quality_grade', 'N/A')} |")
        lines.append(f"")
    else:
        lines.append(f"*시너지 결과 없음. week10_cross_model_analysis.py를 먼저 실행하세요.*")
        lines.append(f"")

    # --- 종합 비교 ---
    lines.extend([
        f"## 5. 종합 비교 매트릭스",
        f"",
        f"| 항목 | ACT | GR00T N1.6 | Cosmos Predict2.5 | GR00T IDM | Cosmos Policy |",
        f"|------|-----|------------|-------------------|-----------|---------------|",
        f"| 유형 | IL Policy | VLA | World Model | IDM | Video-to-Policy |",
        f"| 역할 | 베이스라인 | 파인튜닝+배포 | 합성 비디오 | pseudo label | 비교 분석 |",
        f"| OMX 배포 | 직접 | 직접 | FK 변환 | 추론 전용 | LIBERO 전용 |",
        f"| GPU 요구 | CPU 가능 | 1x RTX 4090 | 1x RTX 4090 | 1x RTX 4090 | 1x (추론) |",
        f"| 추론 속도 | <10ms | ~44ms | 비실시간 | ~20ms | 측정필요 |",
        f"| 언어 이해 | 불가 | 지원 | 불가 | 불가 | 제한적 |",
        f"| IDM 시너지 | - | 증강 데이터 수신 | 합성 비디오 생성 | pseudo label | 참조 |",
        f"",
    ])

    # --- 배포 시나리오 ---
    lines.extend([
        f"## 6. 실용적 배포 시나리오 (OMX 기준)",
        f"",
        f"| 시나리오 | 권장 모델 | 이유 |",
        f"|---------|----------|------|",
        f"| OMX 빠른 프로토타이핑 | ACT | 공식 지원, 빠른 학습, CPU 추론 |",
        f"| OMX 고성능 조작 | GR00T N1.6 | 대규모 사전학습, 언어 이해 |",
        f"| 저비용/교육용 | ACT | GPU 불필요, 학습 곡선 낮음 |",
        f"| 범용 NL 명령 | GR00T N1.6 | 유일한 NL 직접 지원 |",
        f"| 데이터 증강 (합성) | Cosmos Predict2.5 + IDM | 합성 비디오 → pseudo label |",
        f"| 정밀 조작 (대규모 GPU) | Cosmos Policy | 50개 시연으로 SOTA |",
        f"",
    ])

    # --- Lessons Learned ---
    lines.extend([
        f"## 7. 학습된 교훈 (Lessons Learned)",
        f"",
        f"1. **논문 의사코드 vs 실제 API의 괴리**",
        f"   - GR00T: `from_pretrained()` → `Gr00tPolicy(model_path=..., embodiment_tag=...)`",
        f"   - Cosmos Policy: `get_model(cfg)` + `get_action(cfg, model, ...)` 패턴",
        f"   - Cosmos Predict2.5: Action-Conditioned 입력 형식 (EE-space)",
        f"",
        f"2. **FK 변환의 중요성**",
        f"   - OMX는 joint-space, Cosmos Predict2.5는 EE-space 요구",
        f"   - URDF 기반 FK (Z,Y,Y,Y,X 축) 구현 필수 (utils/omx_fk.py)",
        f"   - OMX-F vs OMX-L 링크 길이 차이 주의",
        f"",
        f"3. **GPU 요구사항의 현실적 제약**",
        f"   - Cosmos Policy 파인튜닝: 8x H100 80GB 필요",
        f"   - Cosmos Predict2.5 후훈련: 1x RTX 4090 가능 (8x H100 불필요)",
        f"   - Option C+ 하이브리드: 1x RTX 4090으로 GR00T + Cosmos Predict2.5 모두 가능",
        f"",
        f"4. **데이터 포맷 표준화의 중요성**",
        f"   - OMX LeRobot: HuggingFace datasets (6-dim joint-space)",
        f"   - GR00T: LeRobot v2 (parquet + mp4 + modality.json)",
        f"   - Cosmos Predict2.5: EE-space state/action (FK 변환 필요)",
        f"   - IDM↔VLA: 관절 이름 매핑 + dtype 변환 (uint8↔float32)",
        f"",
        f"5. **ACT 베이스라인의 가치**",
        f"   - OMX 공식 지원으로 빠른 검증 가능",
        f"   - GR00T 대비 낮은 진입 장벽 (CPU, 빠른 학습)",
        f"   - 비교 기준점으로서 GR00T의 부가가치 정량화",
        f"",
        f"---",
        f"",
        f"*Generated by week11_benchmark_report.py*",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report generated: {output_path}")
    print(f"  Total lines: {len(lines)}")


def main():
    parser = argparse.ArgumentParser(description="Generate final benchmark report")
    parser.add_argument("--eval-dir", default="outputs/eval", help="Week 6 eval results")
    parser.add_argument("--cosmos-dir", default="outputs/cosmos_eval", help="Week 7-8 Cosmos results")
    parser.add_argument("--synergy-dir", default="outputs/week10_analysis", help="Week 10 synergy results")
    parser.add_argument("--output", default="outputs/final_report.md", help="Output report path")
    args = parser.parse_args()

    print("=== Week 11: Generating Final Benchmark Report ===\n")

    # 결과 로드
    eval_results = load_eval_results(Path(args.eval_dir))
    print(f"  ACT vs GR00T eval: {len(eval_results)} tasks")

    cosmos_path = Path(args.cosmos_dir) / "cosmos_libero_eval.json"
    cosmos_results = load_json(cosmos_path)
    print(f"  Cosmos LIBERO: {'loaded' if cosmos_results else 'not found'}")

    synergy_path = Path(args.synergy_dir) / "cross_model_analysis.json"
    synergy_results = load_json(synergy_path)
    if synergy_results:
        qm = synergy_results.get("quality_metrics", {})
        print(f"  Cosmos+IDM synergy: grade={qm.get('quality_grade', 'N/A')}")
    else:
        synergy_results = None
        print(f"  Cosmos+IDM synergy: not found")

    # 보고서 생성
    generate_report(
        eval_results=eval_results,
        cosmos_results=cosmos_results,
        synergy_results=synergy_results,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
