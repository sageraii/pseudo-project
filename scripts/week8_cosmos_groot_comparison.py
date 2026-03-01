"""
Week 8: Cosmos Policy vs GR00T N1.6 정량적 비교

두 모델의 데이터 포맷 차이, 추론 속도, 성공률, 데이터 요구량을 비교하고
각 모델의 적합 시나리오를 판정합니다.

Project.md Week 8 명세 기반:
  1. 데이터 포맷 차이 분석 (LeRobot v2 vs Cosmos pickle)
  2. GR00T vs Cosmos Policy 메트릭 비교 테이블
  3. 강점/약점 실증 보고서 생성

Usage:
    python scripts/week8_cosmos_groot_comparison.py \
        --groot-eval-dir outputs/eval \
        --cosmos-eval-dir outputs/cosmos_eval \
        --output-dir outputs/week8_comparison
"""

import argparse
import json
from pathlib import Path

import numpy as np


# =============================================================================
# 데이터 포맷 비교
# =============================================================================

def compare_data_formats() -> dict:
    """GR00T LeRobot v2 vs Cosmos Policy 데이터 포맷 차이 분석"""

    groot_format = {
        "name": "GR00T LeRobot v2",
        "observation": {
            "video": "video.cam1: np.float32 [0,1], shape (B, T, H, W, 3)",
            "state": "state.joint1~gripper: np.float32, 개별 joint 키",
            "language": "annotation.task: [[str]]",
        },
        "action": {
            "format": "개별 joint 키 (joint1~gripper)",
            "representation": "joints=RELATIVE, gripper=ABSOLUTE",
            "horizon": 16,
        },
        "storage": "parquet + mp4 + meta/modality.json",
        "config_format": "Python (register_modality_config)",
    }

    cosmos_format = {
        "name": "Cosmos Policy",
        "observation": {
            "video": "primary_image: np.uint8 [0,255], shape (H, W, 3)",
            "state": "proprio: np.float32, flat vector",
            "language": "T5 text embeddings (사전 인코딩 필요)",
        },
        "action": {
            "format": "flat action vector",
            "representation": "latent action encoding (비디오 latent space)",
            "horizon": "모델 내부 결정",
        },
        "storage": "pickle observation dict",
        "config_format": "YAML + PolicyEvalConfig",
    }

    differences = {
        "video_dtype": {
            "groot": "float32 [0.0, 1.0] (정규화)",
            "cosmos": "uint8 [0, 255] (원본 픽셀)",
            "conversion": "groot_video * 255 → cosmos | cosmos_video / 255.0 → groot",
        },
        "state_format": {
            "groot": "개별 joint 키 (state.joint1, state.joint2, ...)",
            "cosmos": "단일 proprio 벡터 (np.array([...]))",
        },
        "language_input": {
            "groot": "원본 텍스트 문자열 (모델 내부 토크나이징)",
            "cosmos": "T5 임베딩 (사전 계산 필요)",
        },
        "action_output": {
            "groot": "dict {joint1: arr, ..., gripper: arr}",
            "cosmos": "flat numpy array + 미래 상태 예측 + 가치 추정",
        },
    }

    return {
        "groot": groot_format,
        "cosmos": cosmos_format,
        "differences": differences,
    }


# =============================================================================
# 평가 결과 로드
# =============================================================================

def load_json(path: Path) -> dict | None:
    """JSON 파일 안전 로드"""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_groot_results(eval_dir: Path) -> dict:
    """GR00T 평가 결과 로드 (Week 6 결과)"""
    results = {}
    for json_file in sorted(eval_dir.glob("eval_*.json")):
        data = load_json(json_file)
        if data:
            task = data.get("task", json_file.stem)
            results[task] = data.get("groot", {})
    return results


def load_cosmos_results(eval_dir: Path) -> dict:
    """Cosmos Policy 평가 결과 로드 (Week 7 결과)"""
    report = load_json(eval_dir / "cosmos_libero_eval.json")
    if report:
        return report
    return {}


# =============================================================================
# 비교 분석
# =============================================================================

def build_comparison_table(groot_results: dict, cosmos_results: dict) -> list[dict]:
    """GR00T vs Cosmos Policy 비교 테이블 생성"""

    # GR00T 메트릭 집계
    groot_success_rates = []
    groot_latencies = []
    for task_data in groot_results.values():
        if isinstance(task_data, dict):
            sr = task_data.get("success_rate")
            lat = task_data.get("avg_latency_ms")
            if sr is not None:
                groot_success_rates.append(sr)
            if lat is not None:
                groot_latencies.append(lat)

    groot_avg_sr = np.mean(groot_success_rates) if groot_success_rates else None
    groot_avg_lat = np.mean(groot_latencies) if groot_latencies else None

    # Cosmos 메트릭 집계
    cosmos_suites = ["libero_spatial", "libero_object", "libero_goal", "libero_long"]
    cosmos_success_rates = []
    for suite in cosmos_suites:
        sr = cosmos_results.get(suite, {}).get("success_rate") if isinstance(
            cosmos_results.get(suite), dict
        ) else cosmos_results.get(suite)
        if sr is not None:
            cosmos_success_rates.append(sr)

    cosmos_avg_sr = np.mean(cosmos_success_rates) if cosmos_success_rates else None

    # 비교 테이블 행 구성
    table = [
        {
            "metric": "모델 타입",
            "groot": "VLA (Vision-Language-Action)",
            "cosmos": "Video-to-Policy",
        },
        {
            "metric": "성공률 (평균)",
            "groot": f"{groot_avg_sr:.1%}" if groot_avg_sr is not None else "실측 필요",
            "cosmos": f"{cosmos_avg_sr:.1%}" if cosmos_avg_sr is not None else "실측 필요",
        },
        {
            "metric": "추론 속도",
            "groot": f"{groot_avg_lat:.1f}ms" if groot_avg_lat is not None else "~44ms (RTX 4090)",
            "cosmos": "측정 필요 (6-10GB VRAM)",
        },
        {
            "metric": "데이터 요구량",
            "groot": "50-100 demos (파인튜닝)",
            "cosmos": "50-200 demos (후훈련)",
        },
        {
            "metric": "파인튜닝 GPU",
            "groot": "1x H100/L40",
            "cosmos": "8x H100 80GB",
        },
        {
            "metric": "미래 예측",
            "groot": "불가",
            "cosmos": "가능 (latent diffusion)",
        },
        {
            "metric": "언어 이해",
            "groot": "자연어 명령 직접 지원",
            "cosmos": "제한적 (T5 임베딩 필요)",
        },
        {
            "metric": "크로스 플랫폼",
            "groot": "지원 (EmbodimentTag)",
            "cosmos": "미지원",
        },
    ]

    return table


def analyze_strengths_weaknesses() -> dict:
    """각 모델의 강점/약점 실증 분석"""

    return {
        "groot_n16": {
            "strengths": [
                "자연어 명령 직접 입력 가능 (VLM 통합)",
                "크로스 임베디먼트: 다양한 로봇에 파인튜닝 용이",
                "단일 GPU (RTX 4090)로 파인튜닝 및 추론 가능",
                "오픈소스 (Isaac-GR00T, Apache 2.0)",
                "OMX 등 새 로봇에 modality_config 하나로 적응",
            ],
            "weaknesses": [
                "Docker 환경 외 설치 복잡 (NeMo, CUDA 12.6+)",
                "추론 ~44ms → 100Hz 제어 루프에는 비동기 필요",
                "사전훈련 데이터: 주로 휴머노이드 (5-DOF 매니퓰레이터 최적화 아님)",
                "미래 상태 예측 불가 (행동만 출력)",
            ],
        },
        "cosmos_policy": {
            "strengths": [
                "LIBERO 벤치마크 SOTA (~98.5%)",
                "소량 데이터(<200 demos)로 높은 성능",
                "행동 + 미래 상태 + 가치 함수 동시 출력",
                "Test-time Planning 지원",
                "비디오 사전훈련 모델의 물리적 이해 활용",
            ],
            "weaknesses": [
                "Docker 전용 환경 (호스트 설치 어려움)",
                "파인튜닝 시 8x H100 80GB 필요 (고비용)",
                "OMX 등 커스텀 로봇 직접 배포 불가 (LIBERO 전용 체크포인트)",
                "크로스 플랫폼 호환성 없음",
                "언어 이해 제한적 (T5 임베딩 사전 계산 필요)",
            ],
        },
    }


def determine_scenarios() -> list[dict]:
    """각 모델 적합 시나리오 판정"""

    return [
        {
            "scenario": "OMX 빠른 프로토타이핑",
            "recommendation": "GR00T N1.6",
            "reason": "단일 GPU 파인튜닝, modality_config로 빠른 적응",
        },
        {
            "scenario": "LIBERO 벤치마크 재현",
            "recommendation": "Cosmos Policy",
            "reason": "사전훈련 체크포인트로 즉시 평가 가능, SOTA 성능",
        },
        {
            "scenario": "자연어 명령 기반 제어",
            "recommendation": "GR00T N1.6",
            "reason": "유일하게 NL 명령 직접 지원",
        },
        {
            "scenario": "소량 데이터 고성능",
            "recommendation": "Cosmos Policy",
            "reason": "50개 시연으로 ~98% 성공률 (8x H100 파인튜닝 가능 시)",
        },
        {
            "scenario": "정책 안전성 사전 검증",
            "recommendation": "Cosmos Policy",
            "reason": "미래 상태 예측 + 가치 함수로 행동 품질 평가 가능",
        },
        {
            "scenario": "다중 로봇 플랫폼 지원",
            "recommendation": "GR00T N1.6",
            "reason": "EmbodimentTag로 다양한 로봇 지원",
        },
    ]


# =============================================================================
# 보고서 생성
# =============================================================================

def generate_report(
    format_comparison: dict,
    comparison_table: list[dict],
    strengths: dict,
    scenarios: list[dict],
    output_dir: Path,
):
    """마크다운 비교 보고서 생성"""

    lines = [
        "# Week 8: Cosmos Policy vs GR00T N1.6 비교 분석",
        "",
        "## 1. 데이터 포맷 차이",
        "",
        "| 항목 | GR00T N1.6 (LeRobot v2) | Cosmos Policy |",
        "|------|------------------------|---------------|",
    ]

    diffs = format_comparison["differences"]
    for key, vals in diffs.items():
        lines.append(f"| {key} | {vals['groot']} | {vals['cosmos']} |")

    lines += [
        "",
        "## 2. 정량적 비교",
        "",
        "| 메트릭 | GR00T N1.6 | Cosmos Policy |",
        "|--------|-----------|---------------|",
    ]

    for row in comparison_table:
        lines.append(f"| {row['metric']} | {row['groot']} | {row['cosmos']} |")

    lines += [
        "",
        "## 3. 강점/약점 분석",
        "",
        "### GR00T N1.6",
        "",
        "**강점:**",
    ]
    for s in strengths["groot_n16"]["strengths"]:
        lines.append(f"- {s}")
    lines += ["", "**약점:**"]
    for w in strengths["groot_n16"]["weaknesses"]:
        lines.append(f"- {w}")

    lines += ["", "### Cosmos Policy", "", "**강점:**"]
    for s in strengths["cosmos_policy"]["strengths"]:
        lines.append(f"- {s}")
    lines += ["", "**약점:**"]
    for w in strengths["cosmos_policy"]["weaknesses"]:
        lines.append(f"- {w}")

    lines += [
        "",
        "## 4. 적합 시나리오",
        "",
        "| 시나리오 | 권장 모델 | 이유 |",
        "|---------|----------|------|",
    ]
    for s in scenarios:
        lines.append(f"| {s['scenario']} | {s['recommendation']} | {s['reason']} |")

    report_text = "\n".join(lines) + "\n"

    report_path = output_dir / "cosmos_groot_comparison.md"
    with open(report_path, "w") as f:
        f.write(report_text)

    return report_path


# =============================================================================
# 터미널 출력
# =============================================================================

def print_comparison_table(table: list[dict]):
    """비교 테이블 터미널 출력"""
    col_w = [20, 30, 30]
    header = f"{'메트릭':<{col_w[0]}} {'GR00T N1.6':>{col_w[1]}} {'Cosmos Policy':>{col_w[2]}}"
    sep = "-" * sum(col_w)

    print(f"\n{'=' * sum(col_w)}")
    print("  Cosmos Policy vs GR00T N1.6 비교")
    print(f"{'=' * sum(col_w)}")
    print(header)
    print(sep)
    for row in table:
        print(f"{row['metric']:<{col_w[0]}} {row['groot']:>{col_w[1]}} {row['cosmos']:>{col_w[2]}}")
    print(f"{'=' * sum(col_w)}")


def main():
    parser = argparse.ArgumentParser(
        description="Week 8: Cosmos Policy vs GR00T N1.6 정량적 비교"
    )
    parser.add_argument(
        "--groot-eval-dir", default="outputs/eval",
        help="GR00T 평가 결과 디렉토리 (Week 6)",
    )
    parser.add_argument(
        "--cosmos-eval-dir", default="outputs/cosmos_eval",
        help="Cosmos Policy 평가 결과 디렉토리 (Week 7)",
    )
    parser.add_argument(
        "--output-dir", default="outputs/week8_comparison",
        help="비교 보고서 출력 디렉토리",
    )
    args = parser.parse_args()

    groot_eval_dir = Path(args.groot_eval_dir)
    cosmos_eval_dir = Path(args.cosmos_eval_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Week 8: Cosmos Policy vs GR00T N1.6 Comparison ===\n")

    # 1. 데이터 포맷 비교
    print("[1/4] 데이터 포맷 차이 분석...")
    format_comparison = compare_data_formats()
    print(f"  차이점: {len(format_comparison['differences'])}개 항목")

    # 2. 평가 결과 로드
    print("\n[2/4] 평가 결과 로드...")
    groot_results = load_groot_results(groot_eval_dir)
    cosmos_results = load_cosmos_results(cosmos_eval_dir)
    print(f"  GR00T: {len(groot_results)} tasks")
    print(f"  Cosmos: {'로드 완료' if cosmos_results else '결과 없음 (플레이스홀더 사용)'}")

    # 3. 비교 테이블 생성
    print("\n[3/4] 비교 분석...")
    comparison_table = build_comparison_table(groot_results, cosmos_results)
    strengths = analyze_strengths_weaknesses()
    scenarios = determine_scenarios()

    print_comparison_table(comparison_table)

    # 4. 보고서 생성
    print("\n[4/4] 보고서 생성...")
    report_path = generate_report(
        format_comparison, comparison_table, strengths, scenarios, output_dir
    )
    print(f"  Report: {report_path}")

    # JSON 결과 저장
    result = {
        "format_comparison": format_comparison,
        "comparison_table": comparison_table,
        "strengths": strengths,
        "scenarios": scenarios,
    }
    json_path = output_dir / "cosmos_groot_comparison.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  JSON: {json_path}")

    print("\n=== Comparison Complete ===")


if __name__ == "__main__":
    main()
