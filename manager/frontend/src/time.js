import dayjs from "dayjs";
import utc from "dayjs/plugin/utc";

dayjs.extend(utc);

const MOSCOW_OFFSET_MIN = 3 * 60;

export function formatMoscow(ts) {
  if (!ts) return "—";
  return dayjs.utc(ts).utcOffset(MOSCOW_OFFSET_MIN).format("DD.MM.YYYY HH:mm");
}

