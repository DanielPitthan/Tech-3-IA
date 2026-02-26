"""
Eventos de domínio: AlertEvent e ValidationEvent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class AlertEvent:
    """Evento emitido quando um alerta médico é gerado."""

    event_type: str = "alert_generated"
    patient_id: str = ""
    alert_type: str = ""
    severity: str = ""
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationEvent:
    """Evento emitido quando validação humana é requerida."""

    event_type: str = "validation_required"
    patient_id: str = ""
    reason: str = ""
    pending_action: str = ""
    response_draft: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
