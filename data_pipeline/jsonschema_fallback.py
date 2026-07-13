"""Small strict fallback for the checked-in training schema.

The project declares ``jsonschema`` as a runtime dependency.  This module only
keeps data conversion usable in an environment whose requirements have not yet
been refreshed; installed deployments use the upstream package.
"""

from __future__ import annotations

import math


class ValidationError(ValueError):
    def __init__(self, message, absolute_path=()):
        super().__init__(message)
        self.message = str(message)
        self.absolute_path = tuple(absolute_path)


def _matches_type(instance, expected):
    if expected == "object":
        return isinstance(instance, dict)
    if expected == "array":
        return isinstance(instance, list)
    if expected == "string":
        return isinstance(instance, str)
    if expected == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if expected == "number":
        return (
            isinstance(instance, (int, float))
            and not isinstance(instance, bool)
            and math.isfinite(float(instance))
        )
    if expected == "boolean":
        return isinstance(instance, bool)
    if expected == "null":
        return instance is None
    return True


def _validate(instance, schema, path):
    expected_type = schema.get("type")
    if expected_type is not None:
        allowed = expected_type if isinstance(expected_type, list) else [expected_type]
        if not any(_matches_type(instance, candidate) for candidate in allowed):
            raise ValidationError(f"{instance!r} is not of type {expected_type!r}", path)

    if "enum" in schema and instance not in schema["enum"]:
        raise ValidationError(f"{instance!r} is not one of {schema['enum']!r}", path)
    if "minimum" in schema and float(instance) < float(schema["minimum"]):
        raise ValidationError(f"{instance!r} is less than the minimum of {schema['minimum']}", path)
    if "maximum" in schema and float(instance) > float(schema["maximum"]):
        raise ValidationError(f"{instance!r} is greater than the maximum of {schema['maximum']}", path)

    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                raise ValidationError(f"{key!r} is a required property", path)
        properties = schema.get("properties", {})
        for key, value in instance.items():
            if key in properties:
                _validate(value, properties[key], path + (key,))

    if isinstance(instance, list):
        if len(instance) < int(schema.get("minItems", 0)):
            raise ValidationError(
                f"{instance!r} is too short (minimum {schema['minItems']} items)", path
            )
        if "maxItems" in schema and len(instance) > int(schema["maxItems"]):
            raise ValidationError(
                f"{instance!r} is too long (maximum {schema['maxItems']} items)", path
            )
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, value in enumerate(instance):
                _validate(value, item_schema, path + (index,))


def validate(instance, schema):
    _validate(instance, schema, ())

