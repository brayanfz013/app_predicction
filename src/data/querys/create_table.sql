CREATE TABLE position (
    index serial PRIMARY KEY,
    "Fecha Creación" DATE,
    "Material" VARCHAR(50),
    "Material (Cod)" DECIMAL(12,3),
    "Peso neto (TON)" DECIMAL(12,3),
    "Solicitate" VARCHAR(50),
    "Solicitate (Cod)" VARCHAR(15),
    "Unidad Medida Base" VARCHAR(5),
    "Unidad Medida Venta" VARCHAR(5),
    "Cant Pedido UMV" VARCHAR(15),
    "Valor Neto" VARCHAR(15),
    "Contador Ped" VARCHAR(3)
);