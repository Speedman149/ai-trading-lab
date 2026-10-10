"""Validate retrospective action records; apply economics only on their ex-date."""

import math
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from app.data.errors import DataError


class CorporateAction(BaseModel):
    id: str = Field(min_length=1)
    asset: str
    kind: Literal["split", "cash_dividend"]
    ex_date: date
    payable_date: date | None = None
    ratio: float = Field(1, gt=0, allow_inf_nan=False)
    amount: float = Field(0, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def check(self):
        if self.kind == "cash_dividend" and (
            self.amount <= 0 or self.payable_date is None or self.payable_date < self.ex_date
        ):
            raise ValueError("Cash dividends need a positive amount and payable date on/after ex-date")
        return self


def parse_actions(payload, req):
    if not isinstance(payload, dict):
        raise DataError("Alpaca corporate-action response is malformed")
    output = []
    identifiers = set()
    for group, rows in payload.items():
        if not isinstance(rows, list):
            raise DataError("Alpaca corporate-action records must be lists")
        for row in rows:
            if not isinstance(row, dict):
                raise DataError("Malformed corporate-action record")
            # The provider's symbol filter may include both sides of a merger/name change.
            symbols = [value for key, value in row.items() if key == "symbol" or key.endswith("_symbol")]
            if req.asset not in symbols:
                raise DataError("Corporate-action response contains an unexpected symbol")
            effective = row.get("ex_date") or row.get("effective_date") or row.get("process_date")
            try:
                effective = date.fromisoformat(effective)
            except (TypeError, ValueError):
                raise DataError("Corporate-action effective date is missing or invalid") from None
            if not req.start <= effective < req.end:
                continue
            if group not in ("forward_splits", "reverse_splits", "cash_dividends"):
                raise DataError(
                    f"This date range contains an unsupported corporate action: {group}. Select another range."
                )
            if not row.get("ex_date"):
                raise DataError("Split/dividend record has no ex-date")
            identifier = row.get("id")
            if not isinstance(identifier, str) or not identifier or identifier in identifiers:
                raise DataError("Missing or duplicate corporate-action identifier")
            identifiers.add(identifier)
            try:
                values = {"id": identifier, "asset": req.asset, "ex_date": row["ex_date"]}
                if group == "cash_dividends":
                    if (
                        row.get("special")
                        or row.get("foreign")
                        or row.get("due_bill_on_date")
                        or row.get("due_bill_off_date")
                    ):
                        raise DataError(
                            "Special/foreign/due-bill dividends need additional accounting; select another range"
                        )
                    action = CorporateAction(
                        **values,
                        kind="cash_dividend",
                        amount=row["rate"],
                        payable_date=row.get("payable_date"),
                    )
                else:
                    old, new = float(row["old_rate"]), float(row["new_rate"])
                    if not math.isfinite(old) or old <= 0:
                        raise ValueError("Invalid old split rate")
                    action = CorporateAction(**values, kind="split", ratio=new / old)
                    if (group == "forward_splits" and action.ratio <= 1) or (
                        group == "reverse_splits" and action.ratio >= 1
                    ):
                        raise ValueError("Split direction does not match ratio")
                output.append(action)
            except (ValidationError, ValueError, TypeError, KeyError):
                raise DataError("Invalid or incomplete split/dividend record") from None
    by_day = {}
    for action in output:
        by_day.setdefault(action.ex_date, []).append(action.kind)
    if any(len(kinds) > 1 for kinds in by_day.values()):
        raise DataError("Multiple corporate actions on the same ex-date need additional ordering rules")
    return sorted(output, key=lambda action: (action.ex_date, action.id))
