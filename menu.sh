#!/bin/bash


set -e  


function train_model {
  echo "Entrenando el modelo..."
  sudo docker run --rm -v "$(pwd)/train_model:/app/Monosyllables" monosillabic-model bash -c "cd /app/Monosyllables && python train.py && cp monosyllables_model_v2.keras ../monosyllables_model_v0.keras"
  echo "Entrenamiento completado."
}

function build_docker {
  echo "Construyendo la imagen Docker..."
  sudo docker stop monosillabic-api || true
  sudo docker rm monosillabic-api || true
  sudo docker build -t monosillabic-model .
  echo "Imagen Docker construida correctamente."
}

function run_docker {
  echo "Ejecutando el contenedor Docker..."
  sudo docker run -d -p 8125:8125 --name monosillabic-api monosillabic-model
  echo "Contenedor Docker en ejecución."
}

while true; do
  echo ""
  echo "=================================="
  echo " Menú de opciones:"
  echo "1. Entrenar el modelo"
  echo "2. Build Docker"
  echo "3. Run Docker"
  echo "4. Salir"
  echo "=================================="
  read -p "Elija una opción [1-4]: " opcion

  case $opcion in
    1)
      train_model
      ;;
    2)
      build_docker
      ;;
    3)
      run_docker
      ;;
    4)
      echo "Saliendo del menú. ¡Hasta luego!"
      exit 0
      ;;
    *)
      echo "Opción inválida. Intente de nuevo."
      ;;
  esac
done