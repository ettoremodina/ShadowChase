{% macro helper(value) %}
  {% if value %}
    {{ return(value) }}
  {% else %}
    {{ return('fallback') }}
  {% endif %}
{% endmacro %}

{% macro unused_helper() %}
  {{ return('unused') }}
{% endmacro %}

{% macro main(value) %}
  {% set resolved = helper(value) %}
  {% do log(resolved, info=True) %}
  {{ return(resolved) }}
{% endmacro %}
