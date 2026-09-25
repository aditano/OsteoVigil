export type StageId =
  | "solver"
  | "numpy"
  | "scipy"
  | "codecs"
  | "engine"
  | "read"
  | "upload"
  | "decode"
  | "solve"
  | "render";

export type ProgressListener = (label: string, percent: number) => void;

type StageRange = {
  id: StageId;
  start: number;
  end: number;
};

export type RunProgress = {
  advance(stageId: StageId, label: string, fraction?: number): void;
  finish(label?: string): void;
  stop(): void;
};

const USABLE_PERCENT = 99;

export function nextPercent(current: number, target: number): number {
  const rounded = Math.round(target);
  const bounded = Math.min(100, Math.max(0, rounded));
  return Math.max(current, bounded);
}

function spanFor(id: StageId): number {
  switch (id) {
    case "solver":
      return 16;
    case "numpy":
      return 26;
    case "scipy":
      return 10;
    case "codecs":
      return 16;
    case "engine":
      return 8;
    case "read":
      return 8;
    case "upload":
      return 50;
    case "decode":
      return 12;
    case "solve":
      return 12;
    case "render":
      return 5;
    default: {
      const neverId: never = id;
      return neverId;
    }
  }
}

export function stageRanges(stageIds: readonly StageId[]): StageRange[] {
  const spans = stageIds.map((id) => spanFor(id));
  const total = spans.reduce((sum, span) => sum + span, 0);
  let cursor = 0;
  return stageIds.map((id, index) => {
    const width = (spans[index] / total) * USABLE_PERCENT;
    const start = cursor;
    cursor += width;
    const end = index === stageIds.length - 1 ? USABLE_PERCENT : cursor;
    return { id, start, end };
  });
}

export function createRunProgress(stageIds: readonly StageId[], listener: ProgressListener): RunProgress {
  const ranges = new Map(stageRanges(stageIds).map((range) => [range.id, range]));
  const order = new Map(stageIds.map((id, index) => [id, index]));
  let percent = 0;
  let activeIndex = -1;
  let label = "";
  let timer: ReturnType<typeof setInterval> | null = null;
  let creepCeiling = 0;

  function emit(nextLabel: string, target: number, allowComplete = false): void {
    const capped = allowComplete ? target : Math.min(target, USABLE_PERCENT);
    const next = nextPercent(percent, capped);
    if (next === percent && nextLabel === label) {
      return;
    }
    percent = next;
    label = nextLabel;
    listener(label, percent);
  }

  function stopCreep(): void {
    if (timer !== null) {
      clearInterval(timer);
      timer = null;
    }
  }

  function startCreep(ceiling: number): void {
    stopCreep();
    creepCeiling = Math.min(USABLE_PERCENT, Math.max(percent, ceiling));
    if (creepCeiling <= percent) {
      return;
    }
    timer = setInterval(() => {
      if (percent >= creepCeiling) {
        stopCreep();
        return;
      }
      emit(label, percent + 1);
    }, 450);
  }

  return {
    advance(stageId, nextLabel, fraction) {
      const range = ranges.get(stageId);
      const position = order.get(stageId);
      if (!range || position === undefined || position < activeIndex) {
        return;
      }
      activeIndex = position;
      if (fraction === undefined) {
        emit(nextLabel, range.start);
        startCreep(Math.floor(range.end) - 1);
        return;
      }
      stopCreep();
      const clamped = Math.max(0, Math.min(1, fraction));
      const target = range.start + (range.end - range.start) * clamped;
      emit(nextLabel, target);
    },
    finish(nextLabel = "Analysis finished.") {
      stopCreep();
      activeIndex = stageIds.length;
      emit(nextLabel, 100, true);
    },
    stop() {
      stopCreep();
    },
  };
}
